# Genetic Collager
This is a python 2 script that takes an image and uses a genetic algorithm to evolve a closer and closer representation of it over time using small squares from other images, much like scrapbooking or collaging. We start with a random arrangement of "scraps" in random locations, and all of the children slightly change some of these representations. We then compare the image drawn from these dots to the original, and the one with the smallest difference survives. 

# Running 
The code is stored in a single python script. The script will run from the beginning with no arguments, but it expects to find a file named "reference.png" in the same directory that it is running from. It will then create a results/ folder, and output the best performing image every 50 generations.
```bash
python pyointillism.py [-t threads] [-s save file]
```
    -t                               number of threads to use for processing (defaults to all but 1)
    -s                               location of save file to start from
    
