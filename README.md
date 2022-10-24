# Graph Representation for Neuroscience Dataset

## Step

1. Install Dependency:

...

...

2. Download Dataset:

open `utils.py`

run inside the `utils.py`
```
if __name__ == '__main__':
    args = Args().get_args()
    GetData(args).getSubjectData(1)
    pass
```

3. Calculate the Weighted Adjecency Matrix and Train the NN all in

To train Dataset from Subject-1, fold 1, run inside `main.py`
```
if __name__ == '__main__':
    framework = Main()
    framework.fit(subject=1, fold=1)
```
