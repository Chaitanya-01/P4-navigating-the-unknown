# Navigating the unknown -

(Check the full problem statements here [project 4](https://rbe549.github.io/rbe595/fall2023/proj/p4/)


## Steps to run the code
- Install Numpy, OpenCV, djitellopy, torch, cudatoolkit, matplotlib libraries before running the code.
- Install all the library dependencies mentioned [here](https://github.com/princeton-vl/RAFT)
- Turn the drone on and connect to it.
- To run the main code run the `Wrapper.py` file after installing all dependancies. This will save the final output in `Code` folder itself.
- In Code folder:
  ```bash
  python3 Wrapper.py --model=RAFT/models/raft-sintel.pth
  ```
- In our testing we found the weights for sintel dataset are giving better results. Try other weights if you want to by changing the weight file accordingly.


## Collaborators
Chaitanya Sriram Gaddipati - cgaddipati@wpi.edu

Shiva Surya Lolla - slolla@wpi.edu

Ankit Talele - amtalele@wpi.edu




  
