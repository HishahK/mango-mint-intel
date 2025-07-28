import streamlit as st
import os
import logging
import requests
import json
import schedule
import time
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
from bs4 import BeautifulSoup

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_FILE = "qa_cache.json"
COMMON_QUESTIONS_CACHE = {}

def load_cache():
    global COMMON_QUESTIONS_CACHE
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                COMMON_QUESTIONS_CACHE = json.load(f)
    except Exception as e:
        logging.error(f"Error loading cache: {e}")

def save_cache():
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(COMMON_QUESTIONS_CACHE, f)
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

def clear_cache():
    global COMMON_QUESTIONS_CACHE
    COMMON_QUESTIONS_CACHE = {}
    save_cache()

def get_cached_answer(question):
    question_lower = question.lower().strip()
    for cached_q, answer in COMMON_QUESTIONS_CACHE.items():
        if cached_q.lower() in question_lower or question_lower in cached_q.lower():
            return answer
    return None

def cache_answer(question, answer):
    COMMON_QUESTIONS_CACHE[question] = answer
    save_cache()

def load_release_notes_by_source(source):
    if source == "Langflow":
        return load_langflow_releases()
    elif source == "Anthropic":
        return load_anthropic_releases()
    elif source == "ChatGPT":
        return load_chatgpt_releases()
    elif source == "SimpliDOTS":
        return load_simplidots_releases()
    else:
        return []

def load_langflow_releases():
    url = "https://api.github.com/repos/langflow-ai/langflow/releases"
    try:
        response = requests.get(url, headers={'accept': 'application/vnd.github.v3+json'})
        response.raise_for_status()
        releases = response.json()
        return releases
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch Langflow data from GitHub API: {e}")
        return []

def load_anthropic_releases():
    url = "https://docs.anthropic.com/en/release-notes/api"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        releases = []
        
        sections = soup.find_all('section')
        for section in sections:
            heading = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if heading and "release" in heading.get_text().lower():
                release_name = heading.get_text().strip()
                release_body = ""
                for sib in heading.next_siblings:
                    if sib.name and sib.name.startswith('h'):
                        break
                    if sib.string:
                        release_body += sib.get_text(separator="\n").strip() + "\n"
                if release_body.strip():
                    releases.append({"name": release_name, "body": release_body.strip()})
        
        if not releases:
            content_div = soup.find('div', class_='content') or soup.find('article')
            if content_div:
                all_text = content_div.get_text(separator="\n").strip()
                if all_text:
                    releases.append({"name": "Anthropic General Release Notes", "body": all_text[:5000]})

        return releases
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch Anthropic data: {e}")
        return []
    except Exception as e:
        st.error(f"Error parsing Anthropic data: {e}")
        return []

def load_chatgpt_releases():
    url = "https://help.openai.com/en/articles/6825453-chatgpt-release-notes"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        article_body = soup.find('div', class_='article-body')
        if not article_body:
            return []

        releases = []
        headings = article_body.find_all(['h2', 'h3'])
        for heading in headings:
            release_name = heading.get_text().strip()
            release_body = ""
            current_tag = heading.next_sibling
            while current_tag and current_tag.name not in ['h2', 'h3']:
                if current_tag.string:
                    release_body += current_tag.get_text(separator="\n").strip() + "\n"
                current_tag = current_tag.next_sibling
            if release_body.strip():
                releases.append({"name": release_name, "body": release_body.strip()})
        
        if not releases: 
            all_text = article_body.get_text(separator="\n").strip()
            if all_text:
                releases.append({"name": "ChatGPT General Release Notes", "body": all_text[:5000]})

        return releases
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch ChatGPT data: {e}")
        return []
    except Exception as e:
        st.error(f"Error parsing ChatGPT data: {e}")
        return []

def load_simplidots_releases():
    url = "https://fitur-sap.simplidots.id/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        releases = []
        
        post_titles = soup.find_all('h2', class_='entry-title')
        for title_tag in post_titles:
            release_name = title_tag.get_text().strip()
            article_link = title_tag.find('a')['href']
            
            try:
                article_response = requests.get(article_link, timeout=10)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                
                content_body = article_soup.find('div', class_='entry-content')
                if content_body:
                    release_body = content_body.get_text(separator="\n").strip()
                    releases.append({"name": release_name, "body": release_body[:3000]})
                
            except requests.exceptions.RequestException as e:
                logging.warning(f"Could not fetch content for {release_name}: {e}")
            except Exception as e:
                logging.warning(f"Error parsing content for {release_name}: {e}")

        return releases
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch SimpliDOTS data: {e}")
        return []
    except Exception as e:
        st.error(f"Error parsing SimpliDOTS data: {e}")
        return []

@st.cache_resource
def setup_rag_pipeline(source="Langflow", _refresh=False):
    with st.spinner(f"Preparing AI assistant for {source}... (loading data, creating vector store)"):
        releases = load_release_notes_by_source(source)
        if not releases:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_docs = []
        for note in releases:
            if note.get('body'):
                chunks = text_splitter.split_text(note['body'])
                for chunk in chunks:
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={"version": note.get('name', 'unknown'), "source": f"{source} Release {note.get('name', 'unknown')}"}
                    ))
        
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(all_docs, embeddings)

        llm = ChatOpenAI(temperature=0)
        metadata_field_info = [
            AttributeInfo(name="version", description="Release version, e.g., '1.5.0' or 'v0.6.7'", type="string"),
        ]
        document_content_description = "Snippet from release notes"
        
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info,
            verbose=True
        )
        return retriever

def generate_summary(source="Langflow"):
    with st.spinner(f"Creating a summary from the latest 5 {source} releases... This might take a few minutes."):
        releases = load_release_notes_by_source(source)
        if not releases:
            return "No data to summarize."

        content_to_summarize = ""
        for i, release in enumerate(releases[:5]):
            if release.get('body') and len(content_to_summarize) < 10000:
                content_to_summarize += f"\n\n{release['name']}:\n{release['body'][:2000]}"
                if i >= 4:
                    break
        
        if not content_to_summarize.strip():
            return "No content available for summarization."
        
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
        
        prompt = f"""
        You are a technical writer creating an internal summary report for {source}.
        From the following release notes, create one structured final report.
        
        MANDATORY FORMAT TO USE:
        
        ### New Features
        - Point 1 new feature
        - Point 2 new feature
        
        ### Major Changes
        - Point 1 change
        - Point 2 change
        
        ### Important Bug Fixes
        - Point 1 bug fix
        - Point 2 bug fix
        
        If there is no information for a section, write "No significant updates in this section."
        
        RELEASE NOTES:
        {content_to_summarize}
        
        SUMMARY REPORT:
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"

def export_to_pdf(content, filename="summary.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    lines = content.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('###'):
                story.append(Paragraph(line.replace('###', ''), styles['Heading2']))
            elif line.startswith('-'):
                story.append(Paragraph(line, styles['Normal']))
            else:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def send_email_notification(summary, recipient_email, recipient_name="User"):
    api_token = os.getenv("MAILERSEND_API_TOKEN")
    if not api_token:
        logging.error("MAILERSEND_API_TOKEN not set.")
        return False
    try:
        email_data = {
            "from": {
                "email": "noreply@mangomint.dev",
                "name": "Mango Mint Intel"
            },
            "to": [
                {
                    "email": recipient_email,
                    "name": recipient_name
                }
            ],
            "subject": f"Weekly Release Summary - {datetime.now().strftime('%Y-%m-%d')}",
            "html": f"""
            <h2>Hi {recipient_name},</h2>
            <p>Here's your weekly update summary from the release notes:</p>
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 5px; font-family: monospace; white-space: pre-wrap;">
{summary}
            </div>
            <p>Best regards,<br>Mango Mint Intel Bot</p>
            """
        }
        
        response = requests.post(
            "https://api.mailersend.com/v1/email",
            json=email_data,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 202:
            logging.info(f"Email sent successfully to {recipient_email}")
            return True
        else:
            logging.error(f"Failed to send email: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Error sending email: {e}")
        return False

def send_slack_notification(summary, webhook_url):
    try:
        payload = {
            "text": f"Weekly Release Summary - {datetime.now().strftime('%Y-%m-%d')}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Weekly Release Summary*\n```{summary}```"
                    }
                }
            ]
        }
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Error sending Slack notification: {e}")
        return False

def scheduled_summary_generation():
    def generate_and_send():
        summary = generate_summary()
        
        recipient_email = os.getenv("RECIPIENT_EMAIL")
        if recipient_email:
            send_email_notification(summary, recipient_email)
        
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook:
            send_slack_notification(summary, slack_webhook)
        
        logging.info("Scheduled summary generated and sent")
    
    schedule.every().monday.at("09:00").do(generate_and_send)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

load_cache()

st.set_page_config(page_title="Mango Mint Intel", layout="wide")

st.title("Mango Mint Intel: AI Release Assistant")
st.markdown("Smart solution to understand feature updates without reading long texts.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API Key not found. Please create a .env file and add your OPENAI_API_KEY.")
else:
    st.sidebar.header("Settings")
    
    data_source = st.sidebar.selectbox(
        "Select Data Source:",
        ["Langflow", "Anthropic", "ChatGPT", "SimpliDOTS"]
    )
    
    if st.sidebar.button("Refresh Data"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.sidebar.button("Clear Cache"):
        clear_cache()
        st.sidebar.success("Cache cleared successfully!")
    
    st.sidebar.subheader("Email Notification")
    recipient_email = st.sidebar.text_input("Email to receive summaries:", placeholder="example@email.com")
    recipient_name = st.sidebar.text_input("Recipient name:", placeholder="Your Name")
    
    if st.sidebar.button("Test Email Notification"):
        if recipient_email:
            test_summary = """### Test Summary
- Feature 1: Test feature for email notification
- Feature 2: MailerSend integration working

### Bug Fixes 
- Fixed email sending functionality
- Improved notification system"""
            
            if send_email_notification(test_summary, recipient_email, recipient_name or "User"):
                st.sidebar.success("Email sent successfully! Check your inbox.")
            else:
                st.sidebar.error("Failed to send email.")
        else:
            st.sidebar.error("Please enter an email first!")
    
    st.sidebar.subheader("Slack Notification")
    slack_webhook = st.sidebar.text_input("Slack Webhook URL (optional):", value=os.getenv("SLACK_WEBHOOK_URL", ""))
    
    if st.sidebar.button("Test Slack Notification"):
        if slack_webhook:
            test_summary = "Test summary for Slack notification from Mango Mint Intel."
            if send_slack_notification(test_summary, slack_webhook):
                st.sidebar.success("Slack notification sent successfully!")
            else:
                st.sidebar.error("Failed to send Slack notification!")
        else:
            st.sidebar.error("Please enter a Slack Webhook URL first!")
    
    retriever = setup_rag_pipeline(data_source)

    if retriever:
        tab_qa, tab_summary, tab_export = st.tabs(["Q&A", "Automated Summary", "Export & Notifications"])

        with tab_qa:
            st.header("Interactive Q&A")
            st.write(f"Ask specific questions about {data_source} releases here.")
            
            if f"messages_{data_source}" not in st.session_state:
                st.session_state[f"messages_{data_source}"] = [{"role": "assistant", "content": "How can I help you?"}]

            for message in st.session_state[f"messages_{data_source}"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input(f"Example: what are the new features in {data_source}?"):
                st.session_state[f"messages_{data_source}"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            cached_answer = get_cached_answer(prompt)
                            if cached_answer:
                                st.markdown(f"**[From Cache]** {cached_answer}")
                                answer = cached_answer
                            else:
                                context_docs = retriever.invoke(prompt)
                                context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

                                template = "Answer the question `{question}` based only on the following context:\n\n{context}"
                                prompt_template = PromptTemplate.from_template(template)
                                
                                llm = ChatOpenAI()
                                
                                chain = (
                                    {"context": lambda x: context_text, "question": RunnablePassthrough()}
                                    | prompt_template
                                    | llm
                                    | StrOutputParser()
                                )
                                
                                answer = chain.invoke(prompt)
                                st.markdown(answer)
                                cache_answer(prompt, answer)
                            
                            st.session_state[f"messages_{data_source}"].append({"role": "assistant", "content": answer})
                        except Exception as e:
                            logging.error(f"Error during RAG chain execution: {e}")
                            st.error("Sorry, an error occurred while processing your request.")
                            st.session_state[f"messages_{data_source}"].append({"role": "assistant", "content": "Sorry, an error occurred."})

        with tab_summary:
            st.header("Automated Summary Generator")
            st.write(f"Generate a summary from the latest 5 {data_source} release notes.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Summary Now"):
                    if f"summary_{data_source}" in st.session_state:
                        del st.session_state[f"summary_{data_source}"]
                    summary_text = generate_summary(data_source)
                    st.session_state[f"summary_{data_source}"] = summary_text
            
            with col2:
                if st.button("Generate & Send Email"):
                    if not recipient_email:
                        st.error("Please enter a recipient email in the sidebar!")
                    else:
                        summary_text = generate_summary(data_source)
                        st.session_state[f"summary_{data_source}"] = summary_text
                        
                        with st.spinner("Sending email..."):
                            success = send_email_notification(summary_text, recipient_email, recipient_name or "User")
                            if success:
                                st.success(f"Email sent successfully to: {recipient_email}")
                                st.balloons()
                            else:
                                st.error("Failed to send email.")

            if f"summary_{data_source}" in st.session_state and st.session_state[f"summary_{data_source}"]:
                st.success("Summary generated successfully!")
                st.markdown("### Summary Result:")
                st.markdown(st.session_state[f"summary_{data_source}"])

        with tab_export:
            st.header("Export & Notifications")
            
            if f"summary_{data_source}" in st.session_state and st.session_state[f"summary_{data_source}"]:
                st.subheader("Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Download as PDF"):
                        pdf_buffer = export_to_pdf(st.session_state[f"summary_{data_source}"])
                        st.download_button(
                            label="Download PDF",
                            data=pdf_buffer,
                            file_name=f"summary_{data_source}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                
                with col2:
                    if st.button("Download as Markdown"):
                        st.download_button(
                            label="Download Markdown",
                            data=st.session_state[f"summary_{data_source}"],
                            file_name=f"summary_{data_source}_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )
                
                with col3:
                    if st.button("Copy to Clipboard"):
                        st.code(st.session_state[f"summary_{data_source}"], language="markdown")
                        st.info("Copy the text above to your clipboard")
                
                st.subheader("Send Email")
                email_input = st.text_input("Email to send summary:", placeholder="email@example.com")
                name_input = st.text_input("Recipient name:", placeholder="Your Name")
                
                if st.button("Send via Email"):
                    if not email_input:
                        st.error("Please enter an email address!")
                    else:
                        with st.spinner("Sending email..."):
                            success = send_email_notification(st.session_state[f"summary_{data_source}"], email_input, name_input or "User")
                            if success:
                                st.success(f"Email sent successfully to: {email_input}")
                                st.balloons()
                            else:
                                st.error("Failed to send email.")
                
                if slack_webhook:
                    if st.button("Send to Slack"):
                        with st.spinner("Sending to Slack..."):
                            success = send_slack_notification(st.session_state[f"summary_{data_source}"], slack_webhook)
                            if success:
                                st.success("Successfully sent to Slack!")
                            else:
                                st.error("Failed to send to Slack.")
                
                st.subheader("Scheduling")
                st.info("Automatic weekly summaries are scheduled for every Monday at 9:00 AM")
                
                if st.button("Start Background Scheduler"):
                    scheduler_thread = threading.Thread(target=scheduled_summary_generation, daemon=True)
                    scheduler_thread.start()
                    st.success("Background scheduler started! Weekly summaries will be sent automatically.")
            
            else:
                st.info("Generate a summary first to access export options.")
    
    else:
        st.warning("RAG pipeline could not be initialized. Check connection and logs.")