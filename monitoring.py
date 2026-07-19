"""
Script untuk monitoring overfitting & bias during model training/retraining
Gunakan sebelum deploy ke production
"""

import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from collections import defaultdict

KELAS = ['segar', 'layu', 'busuk']
MODEL_PATH = 'model_svm.pkl'
SCALER_PATH = 'scaler.pkl'


class OverfittingMonitor:
    """Monitor untuk deteksi overfitting & bias"""
    
    THRESHOLDS = {
        'overfitting_gap': 0.05,  # Train-Test gap > 5% = overfitting
        'min_recall': 0.85,       # Recall < 85% = underperform
        'max_recall_diff': 0.10,  # Diff antar class > 10% = bias
        'min_f1': 0.85,           # F1 < 85% = poor
    }
    
    def __init__(self):
        self.results = defaultdict(dict)
        self.warnings = []
        self.errors = []
    
    def check_overfitting(self, y_train, y_pred_train, y_test, y_pred_test):
        """Check train-test gap"""
        train_acc = np.mean(y_pred_train == y_train)
        test_acc = np.mean(y_pred_test == y_test)
        gap = train_acc - test_acc
        
        self.results['accuracy'] = {
            'train': train_acc,
            'test': test_acc,
            'gap': gap
        }
        
        print("\n[CHECK 1] Overfitting Detection")
        print("=" * 60)
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy:  {test_acc*100:.2f}%")
        print(f"Gap:            {gap*100:.2f}%")
        
        if gap > self.THRESHOLDS['overfitting_gap']:
            msg = f"⚠️  OVERFITTING: Gap {gap*100:.2f}% > {self.THRESHOLDS['overfitting_gap']*100}%"
            self.warnings.append(msg)
            print(f"   {msg}")
        else:
            print(f"   ✅ Gap OK (<{self.THRESHOLDS['overfitting_gap']*100}%)")
        
        return test_acc, gap
    
    def check_per_class_performance(self, y_test, y_pred_test):
        """Check per-class recall, precision, F1"""
        cm = confusion_matrix(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, 
                                    target_names=KELAS, output_dict=True)
        
        print("\n[CHECK 2] Per-Class Performance")
        print("=" * 60)
        print(f"{'Class':<10} {'Recall':<10} {'Precision':<12} {'F1-Score':<10} {'Status':<15}")
        print("-" * 60)
        
        recalls = []
        f1_scores = []
        
        for i, kelas in enumerate(KELAS):
            recall = report[kelas]['recall']
            precision = report[kelas]['precision']
            f1 = report[kelas]['f1-score']
            
            recalls.append(recall)
            f1_scores.append(f1)
            
            # Status
            if recall >= 0.95 and f1 >= 0.95:
                status = "✅ Excellent"
            elif recall >= self.THRESHOLDS['min_recall'] and f1 >= self.THRESHOLDS['min_f1']:
                status = "✅ Good"
            else:
                status = "❌ Poor"
            
            print(f"{kelas:<10} {recall:<10.4f} {precision:<12.4f} {f1:<10.4f} {status:<15}")
            
            if f1 < self.THRESHOLDS['min_f1']:
                msg = f"⚠️  {kelas.upper()} F1-Score {f1:.4f} < {self.THRESHOLDS['min_f1']}"
                self.warnings.append(msg)
        
        # Check recall balance
        max_recall = max(recalls)
        min_recall = min(recalls)
        recall_diff = max_recall - min_recall
        
        print(f"\n{'Recall Diff':<10} {recall_diff:<10.4f} {'(max-min)':<12} {'':<10}")
        
        if recall_diff > self.THRESHOLDS['max_recall_diff']:
            msg = f"⚠️  CLASS BIAS: Recall diff {recall_diff:.4f} > {self.THRESHOLDS['max_recall_diff']}"
            self.warnings.append(msg)
            print(f"   {msg}")
        else:
            print(f"   ✅ Recall balance OK")
        
        # Confusion Matrix summary
        print(f"\nConfusion Matrix:")
        print(cm)
        
        misclassifications = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    misclassifications.append((KELAS[i], KELAS[j], cm[i, j]))
        
        if misclassifications:
            print(f"\nMisclassifications:")
            for true, pred, count in sorted(misclassifications, key=lambda x: x[2], reverse=True):
                print(f"   - {true.upper()} → {pred.upper()}: {count} times")
        
        return report, cm, recalls
    
    def check_feature_scale(self, X_train_scaled):
        """Check if features are properly scaled"""
        print("\n[CHECK 3] Feature Scaling")
        print("=" * 60)
        
        means = np.mean(X_train_scaled, axis=0)
        stds = np.std(X_train_scaled, axis=0)
        
        mean_close_zero = np.abs(means).max() < 0.1
        std_close_one = np.abs(stds - 1).max() < 0.1
        
        print(f"Feature means max absolute value: {np.abs(means).max():.4f}")
        print(f"Feature stds max deviation from 1: {np.abs(stds - 1).max():.4f}")
        
        if mean_close_zero and std_close_one:
            print("✅ Features properly scaled")
        else:
            msg = "⚠️  Features not properly scaled"
            self.warnings.append(msg)
            print(f"   {msg}")
    
    def generate_report(self, test_acc, gap, report, cm, recalls):
        """Generate final report"""
        print("\n" + "=" * 60)
        print("FINAL ASSESSMENT")
        print("=" * 60)
        
        overall_score = 0
        
        # Scoring
        if gap < 0.02:
            overall_score += 30
            print("✅ [30/100] No overfitting (gap <2%)")
        elif gap < 0.05:
            overall_score += 25
            print("✅ [25/100] Minimal overfitting (gap <5%)")
        else:
            overall_score += 15
            print("⚠️  [15/100] Possible overfitting (gap >5%)")
        
        # Accuracy
        if test_acc >= 0.95:
            overall_score += 30
            print("✅ [30/100] Excellent accuracy (≥95%)")
        elif test_acc >= 0.90:
            overall_score += 25
            print("✅ [25/100] Good accuracy (≥90%)")
        else:
            overall_score += 15
            print("⚠️  [15/100] Acceptable accuracy (<90%)")
        
        # Balance
        recall_diff = max(recalls) - min(recalls)
        if recall_diff < 0.05:
            overall_score += 25
            print("✅ [25/100] Very balanced recall (<5% diff)")
        elif recall_diff < 0.10:
            overall_score += 20
            print("✅ [20/100] Balanced recall (<10% diff)")
        else:
            overall_score += 10
            print("⚠️  [10/100] Unbalanced recall (>10% diff)")
        
        # Per-class performance
        f1_scores = [report[kelas]['f1-score'] for kelas in KELAS]
        if min(f1_scores) >= 0.95:
            overall_score += 15
            print("✅ [15/100] All classes excellent F1 (≥95%)")
        elif min(f1_scores) >= 0.85:
            overall_score += 12
            print("✅ [12/100] All classes good F1 (≥85%)")
        else:
            overall_score += 8
            print("⚠️  [8/100] Some classes poor F1 (<85%)")
        
        print(f"\n{'TOTAL SCORE: ' + str(overall_score) + '/100'}")
        print("=" * 60)
        
        # Recommendation
        if overall_score >= 90:
            status = "🟢 EXCELLENT - Production Ready"
        elif overall_score >= 80:
            status = "🟡 GOOD - Can Deploy with Monitoring"
        elif overall_score >= 70:
            status = "🟠 ACCEPTABLE - Recommend Improvements"
        else:
            status = "🔴 POOR - Do Not Deploy"
        
        print(f"STATUS: {status}")
        
        # Warnings & Errors
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"   - {w}")
        else:
            print("\n✅ No warnings")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"   - {e}")
        
        return overall_score, status


def monitor_training(y_train, y_pred_train, y_test, y_pred_test, X_train_scaled):
    """Wrapper untuk monitoring lengkap"""
    monitor = OverfittingMonitor()
    
    # Check 1: Overfitting
    test_acc, gap = monitor.check_overfitting(y_train, y_pred_train, y_test, y_pred_test)
    
    # Check 2: Per-class
    report, cm, recalls = monitor.check_per_class_performance(y_test, y_pred_test)
    
    # Check 3: Feature scaling
    monitor.check_feature_scale(X_train_scaled)
    
    # Final report
    score, status = monitor.generate_report(test_acc, gap, report, cm, recalls)
    
    return score, status, monitor


# Example usage (jika run standalone)
if __name__ == '__main__':
    print("OverfittingMonitor loaded. Import dan gunakan:")
    print("  from monitoring import monitor_training")
    print("  score, status, monitor = monitor_training(y_train, y_pred_train, y_test, y_pred_test, X_train_scaled)")
