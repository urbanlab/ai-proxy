from peewee import *
from datetime import datetime
import os


os.makedirs('/data', exist_ok=True)

db = SqliteDatabase('/data/requests.db')

class BaseModel(Model):
    class Meta:
        database = db

class Models(BaseModel):
    id = AutoField()
    name = CharField()
    input_cost = FloatField()
    output_cost = FloatField()
    total_cost = FloatField()
    last_reset_date = DateTimeField()

class Users(BaseModel):
    id = AutoField()
    name = CharField()
    input_cost = FloatField()
    output_cost = FloatField()
    total_cost = FloatField()
    last_reset_date = DateTimeField()
   
    
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
    db.create_tables([Requests,Users,Models])
    db.close()

def get_model(model_name):
    return Models.get(Models.name == model_name)

    
def create_request(
        user_name,
        model_name,
        prompt,
        response,
        co2,
        tokens_used=None,
        response_latency=None,
        input_cost = 0,
        output_cost = 0
):
       # search if user exists - get_or_create returns a tuple (instance, created)
    user, created = Users.get_or_create(
        name=user_name,
        defaults={
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost+output_cost,
            'last_reset_date': datetime.now()
            
        }
    )

    model, created = Models.get_or_create(
        name=model_name,
        defaults={
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost+output_cost,
            'last_reset_date': datetime.now()
        }
    )

    # update user cost
    if(user.last_reset_date.month == datetime.now().month):
        user.input_cost = 0
        user.output_cost = 0
        user.total_cost = 0
    else:
        user.input_cost += input_cost
        user.output_cost += output_cost
        user.total_cost += (input_cost + output_cost)
    user.save()
    
    if(model.last_reset_date.month == datetime.now().month):

        model.input_cost = 0
        model.output_cost = 0
        model.total_cost = 0

    else:
        # update user cost
        model.input_cost += input_cost
        model.output_cost += output_cost
        model.total_cost += (input_cost + output_cost)

    model.save()
    
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

