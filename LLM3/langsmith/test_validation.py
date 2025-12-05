from pydantic import BaseModel, Field
class User(BaseModel):
    name : str
    age : int

# user = User(name='홍길동', age=10)
# print(user)
# print(user.name)

class User2():
    def __init__(self, name:str, age:int):
        print(name, age)

user2 = User2('홍길동', '10살')
