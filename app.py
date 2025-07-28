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
   logging.info("Mengambil data rilis Langflow dari GitHub API...")
   url = "https://api.github.com/repos/langflow-ai/langflow/releases"
   try:
       response = requests.get(url, headers={'accept': 'application/vnd.github.v3+json'})
       response.raise_for_status()
       releases = response.json()
       logging.info(f"Berhasil menemukan {len(releases)} rilis Langflow.")
       return releases
   except requests.exceptions.RequestException as e:
       st.error(f"Gagal mengambil data Langflow dari GitHub API: {e}")
       return []

def load_anthropic_releases():
   logging.info("Mengambil data rilis Anthropic...")
   try:
       mock_releases = [
           {"name": "API v2.1", "body": "Introduced Claude 3.5 Sonnet with improved reasoning capabilities and vision features."},
           {"name": "API v2.0", "body": "Major update with Claude 3 family models including Opus, Sonnet, and Haiku variants."}
       ]
       return mock_releases
   except Exception as e:
       st.error(f"Gagal mengambil data Anthropic: {e}")
       return []

def load_chatgpt_releases():
   logging.info("Mengambil data rilis ChatGPT...")
   try:
       mock_releases = [
           {"name": "GPT-4 Turbo", "body": "Enhanced GPT-4 with improved speed and reduced costs. Better instruction following."},
           {"name": "GPT-4 Vision", "body": "Added vision capabilities to GPT-4. Can now analyze images and provide detailed descriptions."}
       ]
       return mock_releases
   except Exception as e:
       st.error(f"Gagal mengambil data ChatGPT: {e}")
       return []

def load_simplidots_releases():
   logging.info("Mengambil data rilis SimpliDOTS...")
   try:
       mock_releases = [
           {"name": "v3.2.1", "body": "Improved dashboard performance and added new analytics features for better user insights."},
           {"name": "v3.2.0", "body": "Major UI overhaul with modern design and enhanced user experience across all modules."}
       ]
       return mock_releases
   except Exception as e:
       st.error(f"Gagal mengambil data SimpliDOTS: {e}")
       return []

@st.cache_resource
def setup_rag_pipeline(source="Langflow", _refresh=False):
   with st.spinner(f"Mempersiapkan asisten AI untuk {source}... (memuat data, membuat vector store)"):
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
       
       logging.info(f"Membuat vector store dari {len(all_docs)} dokumen chunks...")
       embeddings = OpenAIEmbeddings()
       vector_store = Chroma.from_documents(all_docs, embeddings)
       logging.info("Vector store Chroma berhasil dibuat.")

       llm = ChatOpenAI(temperature=0)
       metadata_field_info = [
           AttributeInfo(name="version", description="Versi rilis, contohnya '1.5.0' atau 'v0.6.7'", type="string"),
       ]
       document_content_description = "Potongan dari catatan rilis (release notes)"
       
       retriever = SelfQueryRetriever.from_llm(
           llm,
           vector_store,
           document_content_description,
           metadata_field_info,
           verbose=True
       )
       logging.info("Self-Query Retriever siap digunakan.")
       return retriever

def generate_summary(source="Langflow"):
   with st.spinner(f"Membuat ringkasan dari 5 rilis terbaru {source}... Ini mungkin memakan waktu beberapa menit."):
       releases = load_release_notes_by_source(source)
       if not releases:
           return "Tidak ada data untuk diringkas."

       content_to_summarize = ""
       for i, release in enumerate(releases[:5]):
           if release.get('body') and len(content_to_summarize) < 10000:
               content_to_summarize += f"\n\n{release['name']}:\n{release['body'][:2000]}"
               if i >= 4:
                   break
       
       if not content_to_summarize.strip():
           return "Tidak ada konten yang bisa diringkas."
       
       llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
       
       prompt = f"""
       Anda adalah seorang penulis teknis yang membuat laporan ringkasan internal untuk {source}.
       Dari catatan rilis berikut, buatlah satu laporan akhir yang terstruktur.
       
       WAJIB GUNAKAN FORMAT BERIKUT:
       
       ### Fitur Baru
       - Poin 1 fitur baru
       - Poin 2 fitur baru
       
       ### Perubahan Besar
       - Poin 1 perubahan
       - Poin 2 perubahan
       
       ### Perbaikan Bug Penting
       - Poin 1 perbaikan bug
       - Poin 2 perbaikan bug
       
       Jika tidak ada informasi untuk sebuah bagian, tulis "Tidak ada update signifikan pada bagian ini."
       
       CATATAN RILIS:
       {content_to_summarize}
       
       LAPORAN RINGKASAN:
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
   try:
       api_token = "mlsn.17ccb2c69d1ea2b08b19957f702c335d718553ddd44e077f8f457f9a0e1d775e"
       
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
           <p>Berikut ringkasan update mingguan dari release notes:</p>
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

st.title("Mango Mint Intel: Asisten Rilis AI")
st.markdown("Solusi cerdas untuk memahami update fitur tanpa harus membaca teks panjang.")

if not os.getenv("OPENAI_API_KEY"):
   st.error("Kunci API OpenAI tidak ditemukan. Harap buat file .env dan tambahkan OPENAI_API_KEY Anda.")
else:
   st.sidebar.header("Pengaturan")
   
   data_source = st.sidebar.selectbox(
       "Pilih Sumber Data:",
       ["Langflow", "Anthropic", "ChatGPT", "SimpliDOTS"]
   )
   
   if st.sidebar.button("Refresh Data"):
       st.cache_resource.clear()
       st.rerun()
   
   if st.sidebar.button("Clear Cache"):
       clear_cache()
       st.sidebar.success("Cache berhasil dibersihkan!")
   
   st.sidebar.subheader("Notifikasi Email")
   recipient_email = st.sidebar.text_input("Email untuk menerima ringkasan:", placeholder="contoh@email.com")
   recipient_name = st.sidebar.text_input("Nama penerima:", placeholder="Nama Anda")
   
   if st.sidebar.button("Test Email Notification"):
       if recipient_email:
           test_summary = """### Test Summary
- Feature 1: Test feature untuk email notification
- Feature 2: MailerSend integration working

### Bug Fixes  
- Fixed email sending functionality
- Improved notification system"""
           
           if send_email_notification(test_summary, recipient_email, recipient_name or "User"):
               st.sidebar.success("Email berhasil dikirim! Periksa inbox Anda.")
           else:
               st.sidebar.error("Gagal mengirim email.")
       else:
           st.sidebar.error("Silakan masukkan email terlebih dahulu!")
   
   st.sidebar.subheader("Notifikasi Slack")
   slack_webhook = st.sidebar.text_input("Slack Webhook URL (opsional):", value=os.getenv("SLACK_WEBHOOK_URL", ""))
   
   if st.sidebar.button("Test Slack Notification"):
       if slack_webhook:
           test_summary = "Test summary untuk notifikasi Slack dari Mango Mint Intel."
           if send_slack_notification(test_summary, slack_webhook):
               st.sidebar.success("Slack notification berhasil dikirim!")
           else:
               st.sidebar.error("Gagal mengirim Slack notification!")
       else:
           st.sidebar.error("Silakan masukkan Slack Webhook URL terlebih dahulu!")
   
   retriever = setup_rag_pipeline(data_source)

   if retriever:
       tab_qa, tab_summary, tab_export = st.tabs(["Tanya Jawab (Q&A)", "Ringkasan Otomatis", "Export & Notifikasi"])

       with tab_qa:
           st.header("Tanya Jawab Interaktif")
           st.write(f"Ajukan pertanyaan spesifik tentang rilis {data_source} di sini.")
           
           if f"messages_{data_source}" not in st.session_state:
               st.session_state[f"messages_{data_source}"] = [{"role": "assistant", "content": "Ada yang bisa saya bantu?"}]

           for message in st.session_state[f"messages_{data_source}"]:
               with st.chat_message(message["role"]):
                   st.markdown(message["content"])
           
           if prompt := st.chat_input(f"Contoh: apa fitur baru di {data_source}?"):
               st.session_state[f"messages_{data_source}"].append({"role": "user", "content": prompt})
               with st.chat_message("user"):
                   st.markdown(prompt)

               with st.chat_message("assistant"):
                   with st.spinner("Sedang berpikir..."):
                       try:
                           cached_answer = get_cached_answer(prompt)
                           if cached_answer:
                               st.markdown(f"**[Dari Cache]** {cached_answer}")
                               answer = cached_answer
                           else:
                               context_docs = retriever.invoke(prompt)
                               context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

                               template = "Jawab pertanyaan `{question}` hanya berdasarkan konteks berikut:\n\n{context}"
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
                           st.error("Maaf, terjadi kesalahan saat memproses permintaan Anda.")
                           st.session_state[f"messages_{data_source}"].append({"role": "assistant", "content": "Maaf, terjadi kesalahan."})

       with tab_summary:
           st.header("Generator Ringkasan Otomatis")
           st.write(f"Buat ringkasan dari 5 catatan rilis terbaru {data_source}.")
           
           col1, col2 = st.columns(2)
           
           with col1:
               if st.button("Buat Ringkasan Sekarang"):
                   if f"summary_{data_source}" in st.session_state:
                       del st.session_state[f"summary_{data_source}"]
                   summary_text = generate_summary(data_source)
                   st.session_state[f"summary_{data_source}"] = summary_text
           
           with col2:
               if st.button("Generate & Send Email"):
                   if not recipient_email:
                       st.error("Silakan masukkan email penerima di sidebar!")
                   else:
                       summary_text = generate_summary(data_source)
                       st.session_state[f"summary_{data_source}"] = summary_text
                       
                       with st.spinner("Mengirim email..."):
                           success = send_email_notification(summary_text, recipient_email, recipient_name or "User")
                           if success:
                               st.success(f"Email berhasil dikirim ke: {recipient_email}")
                               st.balloons()
                           else:
                               st.error("Gagal mengirim email.")

           if f"summary_{data_source}" in st.session_state and st.session_state[f"summary_{data_source}"]:
               st.success("Ringkasan berhasil dibuat!")
               st.markdown("### Hasil Ringkasan:")
               st.markdown(st.session_state[f"summary_{data_source}"])

       with tab_export:
           st.header("Export & Notifikasi")
           
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
               email_input = st.text_input("Email untuk dikirim ringkasan:", placeholder="email@example.com")
               name_input = st.text_input("Nama penerima:", placeholder="Nama Anda")
               
               if st.button("Kirim via Email"):
                   if not email_input:
                       st.error("Silakan masukkan alamat email!")
                   else:
                       with st.spinner("Mengirim email..."):
                           success = send_email_notification(st.session_state[f"summary_{data_source}"], email_input, name_input or "User")
                           if success:
                               st.success(f"Email berhasil dikirim ke: {email_input}")
                               st.balloons()
                           else:
                               st.error("Gagal mengirim email.")
               
               if slack_webhook:
                   if st.button("Kirim ke Slack"):
                       with st.spinner("Mengirim ke Slack..."):
                           success = send_slack_notification(st.session_state[f"summary_{data_source}"], slack_webhook)
                           if success:
                               st.success("Berhasil mengirim ke Slack!")
                           else:
                               st.error("Gagal mengirim ke Slack.")
               
               st.subheader("Scheduling")
               st.info("Automatic weekly summaries are scheduled for every Monday at 9:00 AM")
               
               if st.button("Start Background Scheduler"):
                   scheduler_thread = threading.Thread(target=scheduled_summary_generation, daemon=True)
                   scheduler_thread.start()
                   st.success("Background scheduler started! Weekly summaries will be sent automatically.")
           
           else:
               st.info("Generate a summary first to access export options.")
   
   else:
       st.warning("Pipeline RAG tidak dapat diinisialisasi. Periksa koneksi dan log.")