# TypedDict
from typing import TypedDict
from pydantic import BaseModel

# 타입 힌트
# class User(TypedDict):
#     name: str
#     age: str

# u: User  = {
#     'name': '홍길동',
#     'age': 25,
# }

# 타입 강제
class User(BaseModel): 
    name: str
    age: str
user = User(name=12345, age=20)
print(type(user))
print(user)