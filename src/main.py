from datasets import trivago

if __name__ == "__main__":
	sets = trivago.get_trivago_datasets([], percentage=0.1)
