# MM-SCS
The replication package of the paper: MM-SCS: Leveraging Multimodal Features to Enhance Smart Contract Code Search


Contact: shi_research@163.com
### Dependences
- python 3.7
- torch == 1.4.0
- transformers == 3.5.0
- numpy == 1.18.2
- pandas==1.5.1
- numpy==1.23.5
- scipy==1.9.3
### Dataset
Please turn to the dataset repo. [Dataset](https://drive.google.com/drive/folders/1_sSYZeq8blsqrtVMuZhZdhWes_omFgNH?usp=drive_link)


We provide 5 datasets:


- `contracts_full.csv`: The complete 470K dataset, including function name, function codes, function tokens (tokenized function codes), and docstring tokens.
- `contracts_multimodalities.csv`: The deduplicated dataset with additional columns (API sequence and graph structure).
- `train.csv`, `test.csv`, `val.csv`: Split data for training, testing, and validation. Split ratio: 8:1:1.

### Model Training
Run `/script/run_training.sh`.
An example of model training settings in the script:
```
# Define arguments
DATA_PATH="./dataset/train.csv"
SAVE_PATH="./mmscs_model.pth"
BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE=3e-5
MAX_LENGTH=128
MARGIN=0.1

# Execute the training script
python train_with_gat.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --margin $MARGIN
```
Then the model is saved in the '--save_path'.
### Evaluation
Run `/script/run_evaluation.sh`.
An example of model training settings in the script:
```
# Define arguments
TEST_DATA_PATH="./dataset/test.csv"
MODEL_PATH="./mmscs_model.pth"
BATCH_SIZE=4
MAX_LENGTH=128

# Execute the evaluation script
python evaluate_model.py \
    --test_data_path $TEST_DATA_PATH \
    --model_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH

```
### Load Model and Generate Example Outputs
When we get the trained model, we can enter queries and conduct searching in our corpus:

By running scripts `/script/run_code_search.sh`.

Example outputs:
```
Enter your query: Burn tokens of an account
Top matching code snippets:
Result 1 - Similarity Score: 0.9828
    function burnTokens(address _addr, uint256 _amount)public onlyOwner {
        require (balances[msg.sender] < _amount);               
        totalRemainSupply += _amount;                           
        balances[_addr] -= _amount;                             
        burnToken(_addr, _amount);                              
        Transfer(_addr, 0x0, _amount);  

Result 2 - Similarity Score: ...
...
```
### Extract CEDG
`./tools/extracCEDG.py` is used to extract CEDG from a Solidity function. You can also use `visualize_cedg()` function to generate the visualized graph. An example:

![image](https://github.com/user-attachments/assets/8281dddd-b10b-470e-acac-d0311511f858)
