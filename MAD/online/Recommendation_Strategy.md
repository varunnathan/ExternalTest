## Goal: Recommend top-N items to User (u)

## Recommendation Strategy

Step 1: Get the baseline features for u

Step 2:
If num_user_interactions < 20:
    a) Get the list of brands (K) that the user has interacted with - if user has
       bought brands, the top-K brands that are similar else brands in {addToCart, pageView}
    b) For each brand, get the corresponding metadata such as item, ontology, etc.
    c) The clicked_epoch is stored for every user-brand pair. If a new pair is encountered,
       use the latest value for the user
    d) Pick the top-N items based on the model's predicted probability
    Note: optimize the following parameters: K

Else:
    a) Get embeddings for u
    b) Pick the top-K brands based on similarity b/w user embedding and brand embedding.
    c) For each brand, get the corresponding metadata such as item, ontology, etc.
    d) The clicked_epoch is stored for every user-brand pair. If a new pair is encountered,
       use the latest value for the user
    e) Pick the top-N items based on the model's predicted probability
    Note: optimize the following parameters: K

## Artifacts needed for recommendation

a) mapping b/w uuid and index for segLT20
b) mapping b/w uuid and new index for segGE20
c) mapping b/w uuid and baseline features
d) mapping b/w uuid and list of interacted brands
e) mapping b/w brand and item_id
f) mapping b/w ontology and item_id
g) mapping b/w item_id and ontology
h) mapping b/w uuid and embeddings for segGE20
i) mapping b/w brand and embeddings for segGE20
j) model servers for model_segGE20 and model_segLE20 waiting for http requests
