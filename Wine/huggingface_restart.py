
import modal

stub = modal.Stub(name = "Daily_HuggingFace_Restart")
image = modal.Image.debian_slim().pip_install(["huggingface_hub"])
@stub.function(image = image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HUGGINGFACE_API_TOKEN"))
def f():
    g()


def g():
    import os
    from huggingface_hub import HfApi
    token = os.environ["HUGGINGFACE_TOKEN"]
    repo_id = "Sleepyp00/WineMonitoring"
    api = HfApi()
    api.pause_space(repo_id=repo_id, token=token)
    api.restart_space(repo_id=repo_id, token=token)

    

if __name__ == "__main__":
    with stub.run():
        f.remote()
