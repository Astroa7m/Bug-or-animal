# Animal Or bug model
from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from torch.utils.data.datapipes.dataframe.dataframe_wrapper import get_item

if __name__ == "__main__":
    # search function to search for images using duckduckgo library
    def search_images(term, max_results=30):
        print(f"Searching for {term}")
        return L(DDGS().images(term, max_results=max_results)).itemgot("image")


    # searching for animal images and grabbing only one image
    # urls = search_images("animal")

    # dest = "animal.jpg"
    # downloading the image
    # download_url(url=urls[0], dest=dest, show_progress=False)
    # im = Image.open(dest)

    # doing the same for bug image
    # download_url(url=search_images("bug")[0], dest="bug.jpg", show_progress=False)

    from time import sleep

    # downloading images of animal and bugs and put them in their corresponding files
    # terms = "animal", "bug"
    path = Path("animal_or_bug")
    # for term in terms:
    #     dest = (path / term)
    #     dest.mkdir(exist_ok=True, parents=True)
    #     download_images(dest, urls=search_images(f"{term} photo"))
    #     sleep(10)
    #     download_images(dest, urls=search_images(f"{term} sun photo"))
    #     sleep(10)
    #     download_images(dest, urls=search_images(f"{term} shade photo"))
    #     sleep(10)
    #     resize_images(path / term, max_size=400, dest=path / term)

    # removing images that cannot be opened

    # failed = verify_images(get_image_files(path))
    # failed.map(Path.unlink)
    # print(len(failed))

    # # training the model
    dls = DataBlock(
        # inputs are images (bugs and animals)-> imageBlock
        # outputs are category (bug or animal) -> categoryBlock
        blocks=(ImageBlock, CategoryBlock),
        # get the items from the path with the help of this function
        # gets its inner files content (images, input)
        get_items=get_image_files,
        # splitting the dataset into portion for training (0.8) and portion for testing (0.2)
        splitter=RandomSplitter(0.2, 55),
        # ensuring that the output (is the file names) parent of get_image_files result (output)
        get_y=parent_label,
        # resizing every image to 192x192 pixels by squishing
        item_tfms=[Resize(192, method="squish")]
        # loading asynchronously different images and feeding it into the model
    ).dataloaders(path)

    print(dls.show_batch(max_n=9))

    # training the model
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)


    # # testing the model
    #
    # # loading a test image
    # test_img = PILImage.create("animal.jpg")
    #
    # # get prediction
    # pred, pred_idx, probs = learn.predict(test_img)
    #
    # # prediction
    # print(f"Prediction: {pred}")
    #
    # # confidence
    # print(f"Confidence {probs[pred_idx]:.4f}")
    #
    # test_img.show(title=str(pred))

    # exporting the model

    learn.export("animal_or_bug_model.pkl")

