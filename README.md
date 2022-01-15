# END-3.0-Session-10-Assignment
Training and evaluating a transformer for language translation using attention mechanism.

Dataset used : [IWSLT2016](https://pytorch.org/text/stable/datasets.html#iwslt2016) for French to English translation

Procedure used for importing/linking dataset :

1. Imported the dataset from torchtext.datasets<br />
   ```
   from torchtext.datasets import IWSLT2016
   ```

2. Now there's 2 problems :
   - The dataset has sequences too long for colab to handle.
   - The dataset itself is too big.
   
   The result : Colab crashes due to huge memory requirement.
   
   Solutions :
   - I have made a helper function to handle the dataset and use samples with source and target sequence lengths < 100.
     ```
     # Filter out samples with large sequence length... otherwise you need too much RAM.. system crash.
     # Dataset too big... read every offset-th sample
     def filter_function(unfiltered_dataset, offset=1):
         output_list = []
         count = 0
         for sample in unfiltered_dataset:
             # print(sample)
             count+= 1
             if count%offset == 0 and len(sample[0].strip().split(' ')) < 100 and len(sample[1].strip().split(' ')) < 100:
                output_list.append(sample)
     return output_list
     ```
  
   - Used batch size of 32 instead of 64.

3. Due to the ureasonable amount of training time for the dataset, used a subset of the dataset for the purpose of the assignment as shown in `filter_function()` (Accuracy affected).
