# Freiheit 🦋

Freiheit (English: Freedom) is an initiative designed to assist blind and visually impaired individuals with their daily tasks, fostering greater independence through the use of large language multimodal models. The main motto of this project is to provide users with full control over their privacy, and to offer the service at zero cost when used at home. The project is built on an underlying architecture using Langchain, which enables compatibility with various models, including other LLaMA 3 models. While the software was initially tested with in-house models such as the LLaVA (LLaMA 3 + vision) model, it has primarily been evaluated using Omni (ChatGPT-4-o) due to computational constraints.

Freiheit incorporates a wide range of real-life use cases, identified through the exploration of various sources. To test the software, a comprehensive and detailed dataset (from supermarkets, medicine stores, clothings and shoe stores, bus stops, metro stations, streets etc.) was created, considering different scenarios and possibilities. These use cases were selected after thorough research involving publications, videos, and other relevant resources.

Freiheit utilizes multimodal models in backend to recognize images or texts and also predicts the distance of the object in front using either a multimodal model or a trained Neural Network. The inferences for each use case are refined through prompt engineering, using datasets specifically created for these applications. Currently, Freiheit employs Omni (ChatGPT-4-o) or LLaVA to generate inferences.

[![Project demo video](/assets/images/demo_poster.png)](https://www.youtube.com/watch?v=JOuQfZIHabc)

**The app with a complete blind people friendly UI having screen recorder support is under development.**

## Functionalities

- **Bus stops**

  - Reading bus Number and its destination
  - Finding the angular position of Bus
  - Reading display Boards for departure time

- **Metro station**

  - Reading direction and names of platforms
  - Reading display for departure time
  - Reading direction of streets for exit

- **Cloths (for clothings, going out & organising the closet)**:

  - color of cloths
  - Read label tags for shopping
  - Pattern on cloths

- **Laundry**

  - cloth color determination
  - correct pair

- **Products (for shopping or at Supermarkets)**

  - Reading Label Tags
  - Reading product name
  - Reading neutrition
  - Reading product expiry date

- **Facing obstacles in street**: To assist people while walking, the system informs them of the distance to objects ahead. This distance prediction is done in two ways:

  - Using information from large language multimodal models.
  - Using a Neural Network named DistanceNN, based on the DINOV2 architecture. DistanceNN was trained with a small dataset of images showing a chair at distances from 40cm to 400cm, with 5cm intervals. The model achieved an % error of (shown in results/03_Evaluation_plot_with_range.jpeg)

    - 4.9% in 40-100cm
    - 3.5% in 100-200cm
    - 5.9% in 200-300cm
    - 3.0% in 300-400cm

    While the model performed well in testing, it needs improvement for everyday use by training on a larger dataset.
    Because of this, there are two separate files for running the app in the two ways mentioned (explained in the "How to Use" section).

  **Currently, users are encouraged to use the first method (inferences from the large language model only with llm_app.py)**.

- The user gets response both in text and audio in the desired language.

## How to use it

### Prerequisites

The project is built and tested on Python 3.12. You also need to have an OpenAI API key or LlaVA running on your local PC.

Create a conda environment. Make a .env file in your project directory.

- Clone this repository.

```bash
git clone https://github.com/Mnlohani/Freiheit
```

- Create a new environment (for example with Anaconda), then activate it.

```bash
conda create -p [YOUR_ENV_NAME] python=3.12
conda activate [YOUR_ENV_NAME]
```

- Install the requirements

```bash
pip install -r requirements.txt
```

- Get an OpenAI key or Download Llama3. If OpenAI, make a file in your working directory and make a variable OPEN_AI_KEY in that. **Remember to include the file in .gitignore**

- For Llama3

```bash
ollama pull llama3
```

- For getting Inferences

1. From LLM model only (**Currently, users are encouraged to use this**)

```bash
streamlit run llm_app.py
```

2. From both the DistanceNN and LLM models

```bash
streamlit run llm_with_nn_app.py
```

For the second option, user needs to download the DINOV2model pretrained weight tensors (model.safetensors) from [HuggingFace dino-base](https://huggingface.co/facebook/dinov2-base/tree/main) in following directory as below.  
data/03_models/DINOv2_HuggingFace/dinov2_model/model.safetensors

# The underlying Architecure

## The general Architecure

**llm_app.py** includes the following architecure.

## ![The general Architecure](assets/images/architecture_1.png)

**llm_with_nn_app.py** uses above underlying for all use cases except one for prediction of distance of object at front which uses distanceNN. Distance prediction has following architecture.
![Architecture for object detection with DistanceNN](assets/images/architecture_2.png)

#### DistanceNN architecure:

![DistanceNN](assets/images/DistanceNN.png)

#### Running the DistanceNN Pipeline

To execute the DistanceNN pipeline, use the following command:

```python
python -m src.models.distanceNN.run_distanceNN_pipeline
```

**Viewing the Results:**  
Logs will be generated in the logs directory for your review. Additionally, logs will be displayed in the console during execution for real-time monitoring.

## Improvements to be done:

- Make it user friendly (handfree and purely voice based) : Under Development with screen reader support.
- Recognising the tables such as neutritions tables came out to be not correct in the exploration. So further improvement for recongising table based sturcture data using RAG.
- Comparison of results with different models.

_Background image of the App by_ [giorgiotrovato](https://unsplash.com/de/@giorgiotrovato)
