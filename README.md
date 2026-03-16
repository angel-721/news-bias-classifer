# News Bias Classifier


## Model overview
- I created a encoder that I can use to embed chunks of an article with. This was done by fine-tuning DistilBERT on my data set
- After the encoded article text chunks(embeddings) were used to train a basic neural network classifier
- I had about a 88% accuracy when it came to this binary classification problem(bias, non-biased)

## Why did I approach the problem this way?
This method isn't perfect, but I chose to chunked label data to create an attention mask that I can use for a granular way to frame this model.
Instead of training on a entire labeled article with something like TF-IDF, I would rather use BERT-based semantic embedding so I can have a more mature way to create a feature space for this problem.

The above method allows me to chunk a article by groups of paragraphs and classify those chunks so I can find the phrases that make the article biased rather than saying the entire article is biased. Of course this would work better if I took extra steps to isolate those bias parts. This allows me to create a model that reads the article as parts and classify the likely hood of each chunk being biased than using that score to detect the bias score of the entire article which is a more mature method compared to what I did in a previous song genre classifier which was just TF-IDF on an entire song.

This even allows the infer script to provide key signal phrases to show which parts of the article were seen as the most biased. Of course this isn't perfect since my dataset isn't massive and up-to-date. My dataset has articles up to 2024, and if a political event happened in afterwards or just wasn't documented much, such an model won't be perfect at finding that discreet bias chunk. Which is why in the app that uses the project, I use LLM enhancements to provide outside context to the model score.

```
Article text -> chunked DistilBERT embedding(encoder)-> attention classifier
```


## API(🤗 Space)
Deployed as an API as Hugging Face Space: [huggingface.co/spaces/angel-721/News-Bias-Classifier](https://huggingface.co/spaces/angel-721/News-Bias-Classifier)

## App
Used by "The Bias Post" project. [bias-post.angelv.dev](https://bias-post.angelv.dev)


## Dataset:
This project uses the **NewsMediaBias-Plus** dataset created by the Vector Institute Research Team.
> Vector Institute Research Team. *NewsMediaBias-Plus: A Multimodal Dataset for Analyzing Media Bias* (2024).  
> Available at: https://huggingface.co/datasets/vector-institute/newsmediabias-plus
