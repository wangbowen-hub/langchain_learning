from langchain.document_loaders import UnstructuredURLLoader
def test():
    loader=UnstructuredURLLoader(
        urls=["https://baidu.com"],
        model="elements",
        strategy="fast",
    )
    docs=loader.load()
    print(docs)
if __name__=="__main__":
    test()