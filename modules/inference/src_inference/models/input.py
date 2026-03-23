from pydantic import BaseModel


class Message(BaseModel):
    day_id: list[str]  
    but_num_business_unit: list[int]
    dpt_num_department: list[int]