from langchain_core.prompts import PromptTemplate

template=PromptTemplate(input_variables=["product","language"],
               template="""
               You are a helpful product review analyst.
               Summarise customer sentiment for {product} in {language}.
               Be concise and professional.
               Summary:""")

filled=template.format(product="Zomato app",language="Simple English")
print(filled)


filled2=template.format(product="Swiggy Instamart",language="bullet points")
print(filled2)