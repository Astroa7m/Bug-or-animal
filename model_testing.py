from fastai.learner import load_learner
from fastai.vision.core import PILImage


learn = load_learner("animal_or_bug_model.pkl")

test_image = PILImage.create("bug 2.jpg")

print(learn.dls.vocab)

pred, _, probs = learn.predict(test_image)

print(f"Prediction: {pred}")
print(f"Confidence: {probs[0] if pred == "animal" else probs[1]}")