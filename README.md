# ImageLookup

This is a fast algorithm that has the following usecase:
- you have an image or a patch
- you have tons of other images (like a database, maybe millions of images)
- run the algo and it will tell you where did your image or patch come from

# How to run it?
- place your image database (images) in `sources`
- place your query images in `queries`
- run the following:
```
python image_lookup.py
```
- enjoy!

# Dependencies
numpy, tensorflow, opencv, tqdm, sklearn

you can of course check other parameters, which are defined in the argparsed in `utils.py`

# Evaluation
the script will print out an accuracy score if a `gt.txt` file is present in the main dir.
The `gt.txt` is a simple text file that looks like this:
```
query source
query_imageX source_imageY
...
```
where query_imageX and source_imageY are the image names for the source and query (without extension) 
# WIP

