# Receipt Extractor

This repo hosts a receipt extractor application with a [Donut model](https://huggingface.co/spaces/naver-clova-ix/donut-base-finetuned-cord-v2) from the Hugging Face Model Hub.

## Prerequisites

Python 3.10 and `pip` installed. See the [Python downloads page](https://www.python.org/downloads/) to learn more.

## Get started

Perform the following steps to run this project.

1. Clone the repository:

   ```bash
   git clone https://github.com/ryuuzake/receipt-extractor.git
   cd receipt-extractor
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Serve your model as an HTTP server. This starts a local server at [http://localhost:3000](http://localhost:3000/), making your model accessible as a web service.
   
   ```bash
   bentoml serve .
   ```

4. Once your Service is ready, you can build and [containerize it with Docker](https://docs.bentoml.com/en/latest/guides/containerization.html), and deploy it in any Docker-compatible environment.

   ```
   bentoml build --containerize .
   ```

For more information, see [Quickstart in the BentoML documentation](https://docs.bentoml.com/en/latest/get-started/quickstart.html).