+++
title = "Multimodal RAG using Ollama in Google Collab"
date = "2025-05-6"
[taxonomies]
tags = ["ollama", "rag", "multimodal", "llava", "langchain", "colab"]
[extra]
title_image = "ollama.png"
no_emoji_toc = true
+++

<!-- <div style="width: 100%; max-width: 800px; margin: 0 auto;">
    <img src="/ollama.png" alt="Ollama Image" style="width: 100%; height: auto; display: block; margin: 2rem auto;" />
</div> -->

## 🧠 Introduction

Yes, you can run **multimodal RAG** with **Ollama's LLaVA model** inside **Google Colab**!  
The catch? It’s _not_ straightforward — most tutorials skip over how to get Ollama working on Colab’s backend (spoiler: it involves `xterm`).  
In this guide, I break it down step by step — from setting up Ollama to building a LangChain-based pipeline that pulls insights from both text and images.  
Multimodal AI just got easier!

---

## ⚙️ What We’re Building

We’re creating a system that can:

- Read documents (PDFs with text, tables, images)
- Extract text and images
- Summarize each modality
- Store them in a multivector vectorstore
- Use a **LLaVA-powered retriever-augmented chatbot** to answer user queries using **visual + textual context**

<img src="/rag_system.png" alt="Architecture Diagram of Multimodal RAG with Ollama" />

---

## 🔧 Setting Up Ollama in Colab

Ollama isn’t natively supported in Google Colab, so you’ll need to:

1. Install `colab-xterm`
2. Launch the terminal
3. Install and run Ollama manually

### Install Ollama and LLaVA model

First install all the required libraries along with langchain

```python
! pip install -Uqqq pip --progress-bar off
! pip install -qqq transformers -U --progress-bar off
! pip install sentence-transformers  langchain langchain-community langchain-huggingface trl datasets pypdf  -qqq --progress-bar off
! pip install torch torchvision -qqq --progress-bar off
! pip install langchain-ollama "ollama==0.4.2" -q
```

Now the colab-xterm

```python
!pip install colab-xterm
%load_ext colabxterm
```

Start the xterm terminal

```python
%xterm
```

Now in the xterm terminal:

```bash
curl https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llava:7b
```

That's it! Ollama is now running on the backend server of Colab.

---

## 📦 Installing Dependencies

Here’s what you’ll need (besides Ollama):

```python
# installing required libraries
!pip install "unstructured[all-docs]" -q
!pip install langchain-chroma -q
!pip install langchain-community -q
!apt-get install poppler-utils -q
!pip install tiktoken

# installing ocr libraries
!apt-get install tesseract-ocr -q
!apt-get install libtesseract-dev -q


# importing the required libraries
import os
import io
import re
import uuid
import base64
import shutil
import requests
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain.chains.llm import LLMChain, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from openai import OpenAI as OpenAI_vLLM
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')
```

---

## 📄 PDF Ingestion + Image & Table Extraction

We use `unstructured` to chunk the data by title and extract tables/images:

```python
from unstructured.partition.pdf import partition_pdf

def extract_pdf_elements(path, fname):
    return partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title"
    )
```

Then categorize:

```python
def categorize_elements(raw_pdf_elements):
    tables, texts = [], []
    for el in raw_pdf_elements:
        if "Table" in str(type(el)):
            tables.append(str(el))
        elif "CompositeElement" in str(type(el)):
            texts.append(str(el))
    return texts, tables
```

Now we setup up path for file , tables etc and also do the text splitting

```python
# File path
folder_path = "./data/"
file_name = "Satellogic Investor Presentation_December 2024.ptx_.pptx.pdf"

# Get elements
raw_pdf_elements = extract_pdf_elements(folder_path, file_name)

# Get text, tables
texts, tables = categorize_elements(raw_pdf_elements)

# Enforce a specific token size for texts
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000, chunk_overlap = 0
)
joined_texts = " ".join(texts)
texts_token = text_splitter.split_text(joined_texts)

print("No of Textual Chunks:", len(texts))
print("No of Table Elements:", len(tables))
print("No of Text Chunks after Tokenization:", len(texts_token))
```

---

## 🖼️ Image Summarization with LLaVA

We encode images in base64 and prompt LLaVA:

```python
def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
```

```python
from langchain_ollama import ChatOllama

llm_client = ChatOllama(model="llava:7b")

def image_summarize(base64_img, prompt):
    message = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_img}"}
    ]
    return llm_client.invoke([HumanMessage(content=message)]).content
```

**Prompt used:**

> You are an assistant summarizing images for optimal retrieval. Write a clear and concise summary that captures all key visuals, stats, and charts.

---

For the complete implementation of sequential image processing and all the code snippets shown above, check out the accompanying Colab notebook.

## 📚 Text + Table Summarization

For tables and optionally text chunks:

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Summarize the following element for retrieval: {element}"
)

summarize_chain = {
    "element": lambda x: x
} | prompt | llm_client | StrOutputParser()
```

check colab for full implementation !

---

## 🧠 Building the Multivector Retriever

### Creating Retriever

The retriever is a crucial part of our system—it indexes _summaries_ (for efficient search) but returns original raw data (texts, tables, images).

```python
def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)
    return retriever

# The vectorstore to use to index the summaries
vectorstore = Chroma(
    collection_name="mm_rag_vectorstore", embedding_function=embeddings, persist_directory="./chroma_db"
)


# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)
```

> Here, we store summaries in a vectorstore (for similarity search) and raw contents in a docstore (for later use). This is the heart of multi-vector retrieval.

LangChain’s `MultiVectorRetriever` lets us store summaries for retrieval but return the original content.

```python
retriever = MultiVectorRetriever(
    vectorstore=Chroma(...),
    docstore=InMemoryStore(),
    id_key="doc_id"
)
```

We use `ChromaDB` to embed and persist the summary vectors.

```python
vectorstore = Chroma(
    collection_name="mm_rag_vectorstore",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

**Note**

> Here we are using `HuggingFaceEmbeddings(model_name='BAAI/bge-large-en')` as `embeddings`, you can refer to any embedding model compatible with LangChain (like OpenAI, SentenceTransformers, etc.).

Now, we create the Retriever

```python
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)
```

This calls the retriever builder with all our summaries and raw contents (including base64 images).

---

## 💬 The Multimodal RAG Chain

Your final RAG chain combines:

- Document retrieval (image + text)
- Formatting into `HumanMessage`
- LLaVA generation

### Handling Base64 Images

We use utility functions to identify, decode, and resize base64-encoded images.

```python
def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

def is_image_data(b64data):
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"GIF8": "gif",
        b"RIFF": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures)
    except Exception:
        return False

def resize_base64_image(base64_string, size=(64, 64)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

```

These functions ensure that image data is safe, recognizable, and standardized in size before being passed to the model.

### Splitting Text and Image Types

We now clean and split retrieved documents into images and text separately.

```python
def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(64, 64))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
```

This preprocessing is crucial for sending input to LLaVA in the right format.

### Creating the Ollama-Powered Multimodal Chain

We now define the RAG pipeline using `ChatOllama` to talk to the local `llava:7b` model.

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

def multi_modal_rag_context_chain(retriever):
    llm = ChatOllama(
        model="llava:7b",
        temperature=0.7,
        max_tokens=1000
    )

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | llm
        | StrOutputParser()
    )
    return chain

```

This wraps retrieval, preprocessing, prompt formatting, and inference into a single composable chain.

### Formatting Prompt for LLaVA

To help LLaVA understand multimodal input, we structure the prompt as a `HumanMessage`.

```python
def img_prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    content_parts = [{
        "type": "text",
        "text": (
            "You are an advanced AI assistant specialized in analyzing pdf/slides/documents...\n\n"
            f"User's question: {data_dict['question']}\n\n"
            f"Information provided:\n{formatted_texts}"
        )
    }]

    for image in data_dict["context"]["images"]:
        content_parts.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image}",
        })

    return [HumanMessage(content=content_parts)]

```

---

## 🧪 Querying the Multimodal RAG System !

Now you can simply ask questions using:

```python
chain_multimodal_context = multi_modal_rag_context_chain(retriever_multi_vector_img)

def query_multimodal_rag(question):
    try:
        return chain_multimodal_context.invoke(question)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

```

### Optional: Visualize Retrieved Context

To see what the retriever fetched before the model answers:

```python
def display_retrieved_context(question):
    retrieved = retriever_multi_vector_img.invoke(question)
    print("Retrieved Context:")
    for doc in retrieved:
        print(doc.page_content if isinstance(doc, Document) else doc)

    context = split_image_text_types(retrieved)
    if context['images']:
        print("\nRetrieved Images:")
        for img in context['images']:
            plt_img_base64(img)
```

### Final Query Example

```python
question = "What id SATELLOGIC’S MISSION"
response = query_multimodal_rag(question)
print(response)
```

**Answer**

> The image appears to be a presentation slide from a company named Satellogic. The slide contains various texts, images, and graphs related to the company's offerings and strategy. Here's a summary of what I can see:
>
> 1.  **Offering Portfolio:** Satellogic offers services in Asset Monitoring, Constellation-as-a-Service, Space Systems, and a Go-to-Market Strategy.
> 2.  **Asset Monitoring:** High-resolution satellite imagery is provided as part of the offering.
> 3.  **Constellation-as-a-Service:** Satellogic offers a dedicated satellite fleet.
> 4.  **Space Systems:** New sensors and hardware are in orbit.
> 5.  **Go-to-Market Strategy:** The company aims to grow its constellation of satellites while continuing to serve Government, D&I customers to help finance the growing constellation.
> 6.  **Industry Leading Capacity:** Satellogic has multiple daily revisits and over 200 satellites, offering weekly world remaps with near-zero marginal cost.
> 7.  **Contracts & Sales:** The company's customers include Government, D&I, SaaS subscriptions, and commercial customers.
> 8.  **Commercial Model & Platform:** Over time, the company expects that Government, D&I will be less than 20% of its revenues as its commercial line of business and SaaS model scales up.
> 9.  **Technology & Economics:** The slide mentions a patented approach as the most capable and affordable option, and there is a comparison chart between Satellogic and Black|Sky, Maxar, Dee-M, Airbus, and other companies in terms of cost per square kilometer, daily capacity, acquisition cost per square kilometer, and constellation capital expenditure (CAPEX) required for daily world remaps.
> 10. **Pros & Cons:** The slide lists pros such as more photons and short exposure time, and cons such as big size and mass and the inability to continuously capture data without limits on the capture capacity due to the volume of data.
>
> Please note that some text is cut off or hidden, so I might not have captured everything accurately.

Try:

- `"What is Satellogic’s mission?"`
- `"What are their commercial applications?"`
- `"What’s the trend in slide 5?"`

---

## 🤯 Final Thoughts

Getting Ollama to run in Colab was a challenge.  
But once set up, the LangChain integration is smooth — and the power of **multimodal retrieval** is clear.

Your Multimodal can now:

- Interpret PDFs with charts
- Understand business metrics from tables
- Pull out mission statements from visuals

All in one query.

---

## What’s Next?

- Swap in OpenAI or fine-tuned models to improve answer quality while working within resource constraints
- Deploy a frontend (e.g., Streamlit or Gradio)

---

## 🔗 References

- [Ollama](https://ollama.ai/)
- [LangChain](https://www.langchain.com/)

---

Thanks for reading!  
Drop your thoughts at [GitHub](https://github.com/yourusername) or [Twitter](https://twitter.com/yourhandle).

Want the code?  
→ [Open in Google Colab](https://colab.research.google.com/drive/1V2nESWcd0NFMXd_4FiLSVSyEB8Hr4QDi)
