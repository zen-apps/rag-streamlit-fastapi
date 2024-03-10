from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader


faiss_data_path = "./fast_api/app/data/db/faiss_data"


def check_file_exists(file_path: str) -> bool:
    """Check if file exists."""
    fp = file_path + "/index.pkl"
    try:
        with open(fp, "r") as file:
            return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking file: {e}")
        return False


def create_embedded_docs_epa_dataset(refresh_data: bool):
    """Create embedded docs for epa dataset."""
    # download this https://data.consumerreports.org/wp-content/uploads/2022/01/Consumer-Reports-Insights-for-More-Reliable-Electric-Vehicles-Jan-2022.pdf
    # and save it as a pdf in data dir

    embeddings = HuggingFaceEmbeddings()
    if refresh_data:
        pdf_path = "./fast_api/app/data/Consumer-Reports-Insights-for-More-Reliable-Electric-Vehicles-Jan-2022.pdf"

        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        print("reading docs")

        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)

        docsearch = FAISS.from_documents(docs, embeddings)
        num_documents = len(docsearch.index_to_docstore_id)
        print(f"Number of documents: {num_documents}")
        try:
            docsearch.save_local(faiss_data_path)
        except Exception as e:
            print(f"Error saving faiss data: {e}")

    else:
        print("Loading faiss data locally")
        docsearch = FAISS.load_local(
            faiss_data_path, embeddings, allow_dangerous_deserialization=True
        )
        num_documents = len(docsearch.index_to_docstore_id)
        print(f"Number of documents: {num_documents}")

    return {"docsearch": docsearch, "num_documents": num_documents}


def run_retrieval_qa(llm):
    "Run Retrieval QA"
    if check_file_exists(faiss_data_path):

        print("Loading embedded docs")
        embedding_output_dict = create_embedded_docs_epa_dataset(refresh_data=False)
    else:
        print("Creating embedded docs")
        embedding_output_dict = create_embedded_docs_epa_dataset(refresh_data=True)

    docsearch = embedding_output_dict["docsearch"]
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    rqa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return rqa


def ai_retrieval_template():
    """AI retrieval template."""
    template = """You are an expert in car performance data.  Do you best to answer the questions, if you dont know, say you dont know.
    Human question: {question}
    Here is the current conversation history: {conversation_history}
    """
    return template
