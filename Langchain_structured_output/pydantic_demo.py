from pydantic import BaseModel, Field, EmailStr
from typing import Annotated, Optional

class Student(BaseModel):

    name: str
    age: int = Field(lt=0, gt=100, default=23, description="A age must be integer or float.")
    email: Optional[EmailStr] = None
    
new_student = {'name':'santosh', 'email': 'santosh@gmail.com'}

student = Student(**new_student)

print(student)