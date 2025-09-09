from peewee import *
import lib.co2 
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
    created_at = DateTimeField(constraints=[SQL('DEFAULT CURRENT_TIMESTAMP')])

def init_db():
    db.connect()
    db.create_tables([Requests])
    db.close()

def create_request(user_name, model_name, prompt, response, co2_params, duration):
    co2_emission = lib.co2.calculate_co2_emission(co2_params['watt'], co2_params['gram_per_kwh'], duration)
    req = Requests.create(
        user_name=user_name,
        model_name=model_name,
        prompt=prompt,
        response=response,
        co2_emission=co2_emission
    )
    return req

init_db()

