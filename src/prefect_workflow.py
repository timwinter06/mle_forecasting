from prefect import flow


@flow(log_prints=True)
def hello_world(name: str = "world", goodbye: bool = False):
    print(f"Hello {name} from Prefect! 🤗")

    if goodbye:
        print(f"Goodbye {name}!")


if __name__ == "__main__":
    # creates a deployment and stays running to monitor for work instructions
    # generated on the server
    # RUN THIS: prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
    hello_world.serve(name="my-first-deployment", tags=["onboarding"], parameters={"goodbye": True}, interval=60)
