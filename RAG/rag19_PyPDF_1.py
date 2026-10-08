from langchain_community.document_loaders import TextLoader, PyPDFLoader


path ='./_data/'
pdf_loader = PyPDFLoader(path + "attention_is_all_your_needs.pdf")
pdf_docs = pdf_loader.load() # PDF 불러오기

print(type(pdf_docs)) #<class 'list'>
print(len(pdf_docs)) #15
print(pdf_docs) #15
# [Document(metadata={ddd, page_content=
print("============================================")
print(pdf_docs[0]) #첫번째 페이지 보기

