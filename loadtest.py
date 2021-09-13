from locust import HttpUser, task, between 
import random
class WebsiteUser(HttpUser): 
    wait_time = between(3, 4) 
    @task(1) 
    def index(self): 
        self.client.get("/") 
    @task(2) 
    def clickemotion(self):
        diary = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        user= [1,2,3,4]
        self.client.get("click-emotion/%d/%d"%(random.choice(user),random.choice(diary))
        )
    @task(1) 
    def recommand(self):
        ran = [1,2,3,4]
        self.client.get("recommand/%d"% random.choice(ran))
         
        
