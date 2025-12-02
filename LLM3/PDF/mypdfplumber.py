from langchain_community.document_loaders import UnstructuredPDFLoader

# loader = PDFPlumberLoader(
loader = UnstructuredPDFLoader(
    "document_table.pdf",
    mode="elements", # 요소별 분리
    strategy="hi_res" # 고해상도 분석                     
)
documents = loader.load()
print(documents)