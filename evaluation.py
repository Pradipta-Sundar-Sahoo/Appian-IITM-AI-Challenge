import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from scipy.stats import pearsonr
import time

class DocumentCategorizerEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_classification(self, true_labels, predicted_labels, categories):
        """Evaluate document classification performance"""
        # Calculate precision, recall, F1 for each category
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, 
            predicted_labels, 
            average='weighted'
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'kappa': kappa
        }
    
    def evaluate_information_extraction(self, true_info, extracted_info, fields):
        """Evaluate information extraction accuracy"""
        field_scores = {}
        for field in fields:
            if field in true_info and field in extracted_info:
                # Calculate exact match
                exact_match = true_info[field] == extracted_info[field]
                # Calculate partial match using string similarity
                similarity = self._calculate_string_similarity(true_info[field], extracted_info[field])
                field_scores[field] = {
                    'exact_match': exact_match,
                    'similarity': similarity
                }
        
        return field_scores
    
    def evaluate_summarization(self, reference_summaries, generated_summary):
        """Evaluate summary quality using ROUGE and BLEU scores"""
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(generated_summary, reference_summaries[0])
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([summary.split() for summary in reference_summaries],
                                 generated_summary.split())
        
        return {
            'rouge_scores': rouge_scores,
            'bleu_score': bleu_score
        }
    
    def evaluate_chat_performance(self, chat_logs):
        """Evaluate chat system performance"""
        response_times = []
        relevance_scores = []
        user_ratings = []
        
        for log in chat_logs:
            # Calculate response time
            response_time = log['response_timestamp'] - log['query_timestamp']
            response_times.append(response_time)
            
            # Calculate relevance score (if available)
            if 'relevance_score' in log:
                relevance_scores.append(log['relevance_score'])
            
            # Collect user ratings (if available)
            if 'user_rating' in log:
                user_ratings.append(log['user_rating'])
        
        metrics = {
            'avg_response_time': np.mean(response_times),
            'median_response_time': np.median(response_times),
            'avg_relevance': np.mean(relevance_scores) if relevance_scores else None,
            'avg_user_rating': np.mean(user_ratings) if user_ratings else None
        }
        
        return metrics
    
    def _calculate_string_similarity(self, str1, str2):
        """Calculate similarity between two strings using Levenshtein distance"""
        from Levenshtein import ratio
        return ratio(str1, str2)
    
    def generate_evaluation_report(self, all_metrics):
        """Generate comprehensive evaluation report"""
        report = {
            'classification_performance': all_metrics['classification'],
            'information_extraction': all_metrics['extraction'],
            'summarization_quality': all_metrics['summarization'],
            'chat_performance': all_metrics['chat']
        }
        
        return report

# Usage Example
def run_evaluation(categorizer, test_dataset):
    evaluator = DocumentCategorizerEvaluator()
    
    # Collect evaluation metrics
    classification_metrics = evaluator.evaluate_classification(
        test_dataset['true_labels'],
        test_dataset['predicted_labels'],
        test_dataset['categories']
    )
    
    extraction_metrics = evaluator.evaluate_information_extraction(
        test_dataset['true_info'],
        test_dataset['extracted_info'],
        test_dataset['fields']
    )
    
    summarization_metrics = evaluator.evaluate_summarization(
        test_dataset['reference_summaries'],
        test_dataset['generated_summary']
    )
    
    chat_metrics = evaluator.evaluate_chat_performance(
        test_dataset['chat_logs']
    )
    
    # Generate final report
    evaluation_report = evaluator.generate_evaluation_report({
        'classification': classification_metrics,
        'extraction': extraction_metrics,
        'summarization': summarization_metrics,
        'chat': chat_metrics
    })
    
    return evaluation_report