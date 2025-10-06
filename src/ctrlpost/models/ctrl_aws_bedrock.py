'''
CtrlAwsBedrockModel class, to be able to invoke the AWS Bedrock LLMs
'''
import json
import time
import random
from typing import Dict, Optional
from botocore.exceptions import ClientError

import boto3
import torch

from src.ctrlpost.models.base_model import BaseTranslationModel
from src.ctrlpost.utils import evaluation


class CtrlAwsBedrockModel(BaseTranslationModel):
    def __init__(
            self,
            model_name: str,
            s3_bucket: str,
            iam_role: str,
            **kwargs
    ):
        """
        Base PyTorch Lightning model for translation and APE tasks.

        Args:
            model_name (str): Hugging Face model name/path
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 5e-5.
            warmup_steps (int, optional): Number of warmup steps for learning rate scheduler. Defaults to 0.
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 0.0.
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)

        # Load pretrained model and tokenizer
        self.tokenizer = None
        self.model = model_name
        self.bedrock_type = kwargs['bedrock_type']
        if self.bedrock_type == 'batch':
            self.bedrock_client = boto3.client('bedrock')
        elif self.bedrock_type == 'demand':
            self.bedrock_client = boto3.client('bedrock-runtime')
        else:
            raise ValueError(f"Invalid bedrock_type: {self.bedrock_type}")
        self.max_tokens = kwargs['max_length']  # For the generation
        self.rm_prompt_at_decoding = True

        # Generation configuration
        self.temperature = kwargs['temperature']
        self.top_p = kwargs['top_p']

        self.bucket_name = s3_bucket
        # "arn:aws:iam::209378968454:role/ctrlpost-bedrock-inference-role"
        self.role = iam_role

        # S3 temporary data config
        self.input_data_config = {
            "s3InputDataConfig": {
                "s3Uri": f"s3://{s3_bucket}/input_data/input_tmp.jsonl",
            }
        }
        self.output_data_config = {
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{s3_bucket}/output_data/",
            }
        }

        self.model_type = 'decoder-only'

        self.lora_finetuning = False

    def forward(self, **batch):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token ids
            attention_mask (torch.Tensor): Attention mask for input
            target_ids (torch.Tensor, optional): Target token ids for training

        Returns:
            Model output
        """
        return NotImplementedError

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None):
        """
        Prediction step for inference.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (Optional[int], optional): Batch index

        Returns:
            Generated predictions
        """

        preds = []
        for input_text in batch['input_text']:

            response = self.process_single_pred(input_text, retry_count=0)

            response_body = json.loads(response['body'].read().decode('utf-8'))
            result = response_body["content"][0]["text"]
            preds.append(result)
            time.sleep(1)

        return preds

    def process_single_pred(self, input_text, max_retries=100):

        for attempt in range(max_retries):
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.model,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 2048,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": input_text
                                }]
                            }
                        ]
                    }),
                    contentType='application/json'
                )

                # Parse the response
                response_body = json.loads(response['body'].read())
                pred_raw = response_body['content'][0]['text']

                # # Extract the translation (same logic as your original code)
                # start_index = pred_raw.find('{')
                # end_index = pred_raw.find('}')
                # if start_index != -1:
                #     pred = pred_raw[start_index:]
                # if end_index != -1:
                #     pred = pred[:end_index + 1]
                # pred = pred.strip('{}')

                return pred_raw

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for record: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Exponential backoff
                else:
                    return None

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step for the model.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data
            batch_idx (int): Batch index
        """
        # outputs = self(**batch)
        # val_loss = outputs.loss

        # Log validation loss
        # self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Generate predictions
        # preds = self.predict_step(batch)

        if self.bedrock_type == 'batch':
            # Skip batch level prediction, wait until validation epoch ends to run the whole dataset at once
            for i in range(len(batch['target_text'])):

                self.val_predictions.append(
                    {
                        'gt_text': batch['target_text'][i],
                        'input_text': batch['input_text'][i],
                        'contrastive': batch['target_text_contrastive'][i],
                        'raw_source_text': batch['raw_source_text'][i],
                        'gt_annotated': batch['target_annotated_text'][i],
                        'gt_annotated_contrastive': batch['target_annotated_text_contrastive'][i]
                                                    if 'target_annotated_text_contrastive' in batch else None,
                        'attribute_label': batch['attribute_labels'][i]
                    }
                )
        elif self.bedrock_type == 'demand':
            for i in range(len(batch['target_text'])):
                response = self.process_single_pred(batch['input_text'][i][:-1])
                if response:
                    self.val_predictions.append(
                        {
                            'preds': response,
                            'gt_text': batch['target_text'][i],
                            'input_text': batch['input_text'][i],
                            'contrastive': batch['target_text_contrastive'][i],
                            'raw_source_text': batch['raw_source_text'][i],
                            'gt_annotated': batch['target_annotated_text'][i],
                            'gt_annotated_contrastive': batch['target_annotated_text_contrastive'][i]
                                                        if 'target_annotated_text_contrastive' in batch else None,
                            'attribute_label': batch['attribute_labels'][i]
                        }
                    )

        else:
            raise ValueError(f"Invalid bedrock_type: {self.bedrock_type}")

        # time.sleep(5)

    def on_test_epoch_end(self):
        """
        Optional method to log learning rate at the end of each test epoch
        """

        if self.bedrock_type == 'batch':
            # Push all the inputs from the predictions to a jsonl file in S3
            with (open('tmp/input_tmp.jsonl', 'w') as f):
                for idx, pred in enumerate(self.val_predictions):
                    # Write each prediction to the file
                    data_object = {
                        "recordID": idx,
                        "modelInput": {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [ {
                                        "type": "text",
                                        "text": pred['input_text'][:-1]
                                    }
                                    ]
                                }
                            ]
                        }
                    }
                    f.write(json.dumps(data_object) + '\n')
            # Upload the input data to S3
            s3 = boto3.client('s3')
            s3.upload_file(
                'tmp/input_tmp.jsonl', self.bucket_name, 'input_data/input_tmp.jsonl')

            # Invoke batch job with the input data
            response = self.bedrock_client.create_model_invocation_job(
                jobName=f"ctrlpost-job-{int(time.time())}",
                roleArn=self.role,
                modelId=self.model,
                inputDataConfig=self.input_data_config,
                outputDataConfig=self.output_data_config
            )

            # Wait for the job to complete
            job_arn = response.get('jobArn')
            # job_arn = "arn:aws:bedrock:ap-southeast-2:209378968454:model-invocation-job/1gxm5lod7lje"
            while True:
                response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
                status = response['status']
                if status == 'Completed':
                    print("Job succeeded")
                    break
                if status in ['Failed', 'Cancelled']:
                    raise RuntimeError(f"Job failed with status {status}")
                time.sleep(60)  # Wait for a minute before checking again

            # Once the job has succeeded, download the output data
            suffix_job = job_arn.split('model-invocation-job/')[-1]
            s3.download_file(self.bucket_name, f"output_data/{suffix_job}/input_tmp.jsonl.out",
                             'tmp/output_tmp.jsonl')

            all_predictions = self.val_predictions

            # Read the output data
            with open('tmp/output_tmp.jsonl', 'r') as f:
                for line in f:
                    response = json.loads(line)

                    record_id = int(response['recordId']) - 1  # Adjust for zero-based index

                    pred = response['modelOutput']['content'][0]['text']
                    all_predictions[record_id]['preds'] = pred

        elif self.bedrock_type == 'demand':
            all_predictions = self.val_predictions

        else:
            raise ValueError(f"Invalid bedrock_type: {self.bedrock_type}")

        prep_all_predictions = self.prepare_predictions(all_predictions)
        dict_scores = evaluation(prep_all_predictions, self.tgt_lng, save_preds=True,
                                 comet_model=self.comet_model, loggers=self.loggers)
        # dict_scores = self.evaluation_scores(all_predictions, save_preds=True)
        for key, value in dict_scores.items():
            self.log(f'{self.eval_log_type}_{key}', value)
            print(f'{self.eval_log_type}_{key}: ', value)

        self.val_predictions = []

    def prepare_predictions(self, all_predictions):
        prediction_list = []
        src_list = []
        src_raw_list = []
        gt_list = []
        gt_annotated_list = []
        contrastive_list = []
        gt_annotated_list_contrastive = []
        attribute_label_list = []
        data_comet = []
        for preds in all_predictions:
            # Decode each sentence (input, prediction, and ground truth)
            src_text = preds['input_text']
            pred_text = preds['preds']
            if hasattr(self, 'rm_prompt_at_decoding'):
                if self.rm_prompt_at_decoding:
                    # Detect the start of the translation with the '{' character
                    start_index = pred_text.find('{')
                    end_index = pred_text.find('}')
                    if start_index != -1:
                        pred_text = pred_text[start_index:]
                    if end_index != -1:
                        pred_text = pred_text[:end_index + 1]
                    pred_text = pred_text.strip('{}')
            # print('\nTGT')
            # print(pred_text[0])
            raw_source_text = preds['raw_source_text']
            src_raw_list.append(raw_source_text)
            gt_text = preds['gt_text']
            gt_annotated_text = preds['gt_annotated']
            attribute_label = preds['attribute_label']
            prediction_list.append(pred_text)
            src_list.append(src_text)
            gt_list.append(gt_text)
            gt_annotated_list.append(gt_annotated_text)
            if preds['gt_annotated_contrastive']:
                contrastive_list.append(preds['contrastive'])
                gt_annotated_text_contrastive = preds['gt_annotated_contrastive']
                gt_annotated_list_contrastive.append(gt_annotated_text_contrastive)
            attribute_label_list.append(attribute_label)
            data_comet.append({'src': raw_source_text, 'mt': pred_text, 'ref': gt_text})

        prep_dict = {
            'prediction_list': prediction_list,
            'src_raw_list': src_raw_list,
            'src_list': src_list,
            'gt_list': gt_list,
            'gt_annotated_list': gt_annotated_list,
            'contrastive_list': contrastive_list,
            'gt_annotated_list_contrastive': gt_annotated_list_contrastive,
            'attribute_label_list': attribute_label_list,
            'data_comet': data_comet
        }

        return prep_dict
