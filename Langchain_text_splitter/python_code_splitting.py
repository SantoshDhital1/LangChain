from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __eq__(self, other):
        return self.title == other.title and self.author == other.author


# Usage
book1 = Book("1984", "George Orwell")
book2 = Book("1984", "George Orwell")
print(book1 == book2)  # Output: True
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)


# perform the split
result = splitter.split_text(text)

print(len(result))

print(result[2])