from locust import HttpUser, task, between

class AbuseDetectorUser(HttpUser):
    wait_time = between(0.1, 0.3)  # 3–10 RPS

    @task
    def language_detect(self):
        self.client.post("/api/v1/language/detect", json={"text": "you are useless"})

    @task
    def check_profanity(self):
        # Define the endpoint and the payload
        endpoint = "/api/v1/moderation/text"
        payload = {"text": "I will kill you", "language": "en"}
        headers = {"accept": "application/json", "Content-Type": "application/json"}

        # Make the POST request
        self.client.post(endpoint, json=payload, headers=headers)