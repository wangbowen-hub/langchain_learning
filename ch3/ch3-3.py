from langchain.prompts import(
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
system_template="你是一个翻译助手，可以将{input_lang}翻译成{output_lang}"
system_prompt=SystemMessagePromptTemplate.from_template(system_template)
human_template="{talk}"
human_prompt=HumanMessagePromptTemplate.from_template(human_template)
chat_prompt=ChatPromptTemplate.from_messages([system_prompt,human_prompt])
messages=chat_prompt.format_messages(input_lang="中文", output_lang="英语", talk="我喜欢编程")
print(messages)
