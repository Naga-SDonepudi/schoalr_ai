import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
COURSE_NAME = os.getenv('COURSE_NAME')
DEVICE = os.getenv('DEVICE', 'cpu')  # Default to 'cpu' if not set

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)
data_dir = os.path.join(parent_dir, "data")
vector_db_dir = os.path.join(parent_dir, "vector_db")
chapters_vector_db_dir = os.path.join(parent_dir, "chapters_vector_db")

embedding = HuggingFaceEmbeddings(model_kwargs={"device": DEVICE}) # Use device from env
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)


def vectorize_book_and_store_to_db(course_name, vector_db_name):
    os.makedirs(vector_db_dir, exist_ok=True)
    book_dir = os.path.join(data_dir, course_name)
    vector_db_path = os.path.join(vector_db_dir, vector_db_name)

    all_documents = []
    for file in os.listdir(book_dir):
        if not file.endswith('.pdf'):
            continue
        pdf_path = os.path.join(book_dir, file)
        loader = UnstructuredLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_chunks = text_splitter.split_documents(all_documents)
    text_chunks = filter_complex_metadata(text_chunks)  # ← add this
    Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory=vector_db_path)
    print(f"{course_name} saved to vector db: {vector_db_name}")
    return 0


def vectorize_chapters(course_name):
    os.makedirs(chapters_vector_db_dir, exist_ok=True)
    book_dir = os.path.join(data_dir, course_name)
    for chapter in os.listdir(book_dir):
        if not chapter.endswith('.pdf'):
            continue
        chapter_name = chapter[:-4]
        chapter_pdf_path = os.path.join(book_dir, chapter)
        loader = UnstructuredLoader(chapter_pdf_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        texts = filter_complex_metadata(texts)  # ← add this
        chroma_path = os.path.join(chapters_vector_db_dir, chapter_name)
        Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=chroma_path)
        print(f"{chapter_name} chapter vectorized")
    return 0