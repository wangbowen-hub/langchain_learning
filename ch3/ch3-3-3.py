import json
import re
from typing import Type,TypeVar
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate
from pydantic import Field,BaseModel
from pydantic_core import ValidationError
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException

CUSTOM_FORMAT_INSTRUCTIONS  = """输出内容需要被格式化成一个 JSON 实例，这个实例应该符合下面提到的 JSON 模式。
输出模式如下：
```
{schema}
```"""
#!!!
T=TypeVar("T",bound=BaseModel)
class CustomOutputParser(BaseOutputParser[BaseModel]):
    #!!!
    pydantic_object:Type[T]
    def parse(self,text:str)->BaseModel:
        try:
            json_pattern = r'\n\`\`\`json(.*?)\`\`\`\n'
            #!!!
            json_match=re.search(json_pattern,text,re.DOTALL)
            if json_match:
                json_content=json_match.group(1)
                python_object=json.loads(json_content,strict=False)
                #!!!
                expense_records=[self.pydantic_object.model_validate(item) for item in python_object]
                return expense_records
        except(json.JSONDecodeError,ValidationError)as e:
            name=self.pydantic_object.model_json_schema()
            msg = f"从输出中解析{name}失败 {text}。错误信息: {e}"
            raise OutputParserException(msg, llm_output=text)
        
    def get_format_instruction(self)->str:
        schema=self.pydantic_object.model_json_schema()
        reduced_schema=schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        schema_str=json.dumps(reduced_schema)
        return CUSTOM_FORMAT_INSTRUCTIONS.format(schema=schema_str)
    
    @property
    def _type(self)->str:
        return "custom output parse"

if __name__=="__main__":
    class ExpenseRecord(BaseModel):
        amount:float=Field(description="花费金额")
        category:str=Field(description="花费类别")
        date:str=Field(description="花费日期")
        description:str=Field(description="花费描述")
    parser=CustomOutputParser(pydantic_object=ExpenseRecord)
    # 定义获取花费记录的提示模板
    expense_template = '''
    请将这些花费记录在我的预算中。
    我的花费记录是：{query}
    格式说明：
    {format_instructions}
    '''
    #!!!
    prompt=PromptTemplate(template=expense_template,input_variables=["query"],partial_variables={"format_instructions":parser.get_format_instructions()})
    model=Tongyi()
    chain=prompt|model
    expense_records = parser.parse(chain.invoke({"query": "昨天,我在超市花了45元买日用品。晚上我又花了20元打车。"}))
    # 遍历并打印花费记录的各个参数
    for expense_record in expense_records:
        print(expense_record.__dict__)

    



