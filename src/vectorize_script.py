import os
from dotenv import load_dotenv
from vectorize_book import vectorize_book_and_store_to_db, vectorize_chapters


load_dotenv()

COURSE_NAME = os.getenv('COURSE_NAME')


vectorize_book_and_store_to_db(
    COURSE_NAME, "machinelearning_and_deeplearning_vectordb"
)
vectorize_chapters(COURSE_NAME)
