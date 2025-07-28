# Mango Mint Intel: AI Release Assistant

## Overview

In the Mango Mint Distribution internal team, release notes and feature updates often go unread or are missed by team members. This happens for various reasons, such as lack of time, forgetting to read them, not receiving notifications, or simply being disinterest in reading lengthy texts.

This project, "Mango Mint Intel," addresses this problem by providing an **AI-powered solution** that fetches **actual feature updates (release notes)**, summarizes them periodically, and offers an interactive chatbot for Q&A. This ensures your team stays informed effortlessly, making essential information easily accessible and digestible.

---

## Key Features

Mango Mint Intel is designed with the following core functionalities:

### 1. **Automated Release Notes Processing & Summarization**
* **Dynamic Data Ingestion**: The system fetches and processes release notes from live web sources. The method of data retrieval varies based on the availability of a dedicated API:
    * **Direct API Access (Langflow)**: For Langflow, a public and well-documented **API endpoint** (GitHub Releases API) is available. Using an API is preferred because it provides structured, reliable data in a machine-readable format (JSON), is generally faster, more stable, and less prone to breaking from website layout changes.
    * **Web Scraping (Anthropic, ChatGPT, SimpliDOTS)**: For Anthropic, ChatGPT, and SimpliDOTS, direct, public APIs specifically for their release notes are not readily available or explicitly provided for third-party consumption. Therefore, the application uses **web scraping** (`requests` and `BeautifulSoup`) to extract information directly from their public web pages. While this offers flexibility to gather data from publicly available content, it is more susceptible to disruptions if the website's HTML structure changes.
* **Automated Weekly/Monthly Recaps**: Generates structured summaries (e.g., in a newsletter format) of the latest release updates. These summaries highlight new features, major changes, and important bug fixes.
* **Export Options**: Summaries can be downloaded in **PDF** or **Markdown** format, or copied directly to the clipboard for easy sharing.

### 2. **Interactive Q&A Chatbot (Retrieval-Augmented Generation - RAG)**
* **Contextual Understanding**: Leverages a **Retrieval-Augmented Generation (RAG)** approach using `Chroma` as a vector store and `OpenAIEmbeddings` to provide accurate and grounded answers to user queries.
* **Self-Query Retriever**: Utilizes LangChain's `SelfQueryRetriever` to intelligently filter and retrieve relevant documents based on user questions, improving answer precision.
* **Chat Interface**: Provides a user-friendly chat interface within the Streamlit application where team members can ask questions like:
    * "What are the new features this month?"
    * "What changes were made to feature X this week?"
* **Caching for Common Questions**: Implements a simple caching mechanism for frequently asked questions to reduce latency and optimize LLM token usage, providing faster responses for common queries.

### 3. **Integration & Notifications**
* **Email Notifications**: Automatically sends generated summaries to specified email recipients via the MailerSend API, ensuring team members receive updates directly in their inboxes.
* **Slack Integration**: Supports sending summary notifications to Slack channels using Webhook URLs, providing another convenient delivery channel.
* **Scheduled Automation**: Includes a background scheduler that can be started to automatically generate and send weekly summaries (e.g., every Monday at 9:00 AM).

### 4. **Cost and Token Optimization**
* **Efficient Summarization**: The summary generation process is optimized to summarize the most recent 5 releases, reducing the amount of context passed to the LLM.
* **Caching**: Reduces redundant LLM calls for common questions.

---

## Setup and Local Execution

Follow these steps to set up and run the Mango Mint Intel project locally:

### Prerequisites

* Python 3.8+
* `pip` package manager
* An **OpenAI API Key**
* **`beautifulsoup4`**: Required for web scraping.
* A MailerSend API Token (optional, for email notifications)
* A Slack Webhook URL (optional, for Slack notifications)

### Installation Steps

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/hishahk/mango-mint-intel.git](https://github.com/hishahk/mango-mint-intel.git)
    cd mango-mint-intel
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure `requirements.txt` includes `beautifulsoup4`, `requests`, `streamlit`, `python-dotenv`, `langchain-openai`, `langchain-community`, `reportlab`, `schedule`)*

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project and add your API keys and optional notification settings:

    ```env
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    MAILERSEND_API_TOKEN="YOUR_MAILERSEND_API_TOKEN" # Optional
    RECIPIENT_EMAIL="your_team_email@example.com" # Optional, for scheduled emails
    SLACK_WEBHOOK_URL="YOUR_SLACK_WEBHOOK_URL" # Optional
    ```
    Replace the placeholder values with your actual keys and URLs. **Do not commit your `.env` file to version control.** An example `.env.example` file is provided for reference.

### Running the Application

1.  **Start the Streamlit Application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the Application:**
    The application will open in your default web browser, usually at `http://localhost:8501`.

---

## Demo Video

A short screen recording demonstrating the functionality of the Mango Mint Intel AI Assistant is available here:

[**Link to Demo Video**](https://youtu.be/PGau6yFH1Ww)

---

## Important Notes for Reviewers

* **API Keys**: Ensure your `OPENAI_API_KEY` is set correctly in the `.env` file for the application to function. MailerSend and Slack API keys are optional but required for their respective notification features.
* **Web Scraping Fragility**: For Anthropic, ChatGPT, and SimpliDOTS, data is fetched via web scraping. This method is inherently less stable than using a dedicated API as it relies on the HTML structure of the source website. If the website's layout changes, the scraping logic might break, requiring updates to the code. A production-grade solution would typically aim for official APIs where available, or more robust scraping frameworks with advanced error handling and monitoring.
* **Caching**: The Q&A feature includes a caching mechanism. The first time you ask a question, it might take slightly longer. Subsequent identical or very similar questions will likely return a cached response, indicated by `[From Cache]`. You can clear the cache from the sidebar.
* **Scheduled Tasks**: The background scheduler for automated summaries runs in a separate thread. While useful for local testing, for production deployments, consider using more robust scheduling solutions like Cron jobs, Airflow, or cloud-based schedulers for reliability.
* **Error Handling**: Basic error handling is implemented for API calls and LLM interactions. If issues occur, check the Streamlit logs for more details.

---

## Future Enhancements (Optional Tasks Implemented)

This project has incorporated several optional tasks to enhance its value:

* **Retrieval-Augmented Generation (RAG)**: Implemented using `Chroma` for vector storage and `SelfQueryRetriever` for accurate contextual answers.
* **Integration with Popular Tools**: Supports **email (MailerSend)** and **Slack** notifications for automated summaries.
* **Caching for Common Questions**: Reduces latency and token usage for repeated queries.
* **Cost and Token Usage Optimization**: Achieved through selective summarization (summarizing top 5 recent releases) and caching.
* **Flexible Release Notes Update**: The design allows for easy integration of new release notes data by simply extending the `load_release_notes_by_source` function, without rebuilding the core system.

---

Feel free to explore the application and provide any feedback!