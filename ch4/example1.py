from langchain_core.runnables import RunnableLambda


if __name__ == "__main__":
    
    def add(x,y):
        return x+y
    def multify(x):
        return 2*x
    runnable_add=RunnableLambda(lambda args:add(**args))
    runnable_multify=RunnableLambda(multify)
    chain=runnable_add|runnable_multify
    print(chain.invoke({"x":2,"y":3}))