from peewee import *
import os


os.makedirs('/data', exist_ok=True)

db = SqliteDatabase('/data/requests.db')

class BaseModel(Model):
    class Meta:
        database = db
    
class Requests(BaseModel):
    id = AutoField()
    user_name = CharField()
    model_name = CharField()
    prompt = TextField()
    response = TextField()
    co2_emission = FloatField()
    tokens_used = IntegerField(null=True) 
    response_latency = FloatField(null=True)
    created_at = DateTimeField(constraints=[SQL('DEFAULT CURRENT_TIMESTAMP')])

def init_db():
    db.connect()
    db.create_tables([Requests])
    db.close()

def create_request(user_name, model_name, prompt, response, co2, tokens_used=None, response_latency=None):
    req = Requests.create(
        user_name=user_name,
        model_name=model_name,
        prompt=prompt,
        response=response,
        co2_emission=co2,
        tokens_used=tokens_used,
        response_latency=response_latency
    )
    return req

init_db()

