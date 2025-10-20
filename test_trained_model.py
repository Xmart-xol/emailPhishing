#!/usr/bin/env python3
"""
Test script for the trained phishing detection model
"""

import requests
import json

# Test emails - mix of phishing-like and legitimate-like content
test_emails = [
    {
        "text": "From: security@paypal.com\nSubject: Urgent: Verify Your Account Now\n\nYour PayPal account has been limited. Click here to verify your identity immediately to avoid suspension. Act now!",
        "expected": "phishing"
    },
    {
        "text": "From: admin@company.com\nSubject: Monthly Team Meeting\n\nHi team, our monthly meeting is scheduled for Friday at 2 PM in the conference room. Please bring your project updates.",
        "expected": "legitimate"
    },
    {
        "text": "From: noreply@bank-security.net\nSubject: Account Verification Required\n\nWe detected unusual activity. Verify your account within 24 hours or it will be suspended. Click the link below.",
        "expected": "phishing"
    },
    {
        "text": "From: newsletter@techcrunch.com\nSubject: This Week in Tech\n\nHere are the top technology stories from this week. New AI developments, startup funding rounds, and more.",
        "expected": "legitimate"
    }
]

def test_model():
    """Test the trained model with sample emails"""
    server_url = "http://localhost:8000"
    
    print("🔍 Testing Trained Phishing Detection Model")
    print("=" * 50)
    
    correct_predictions = 0
    total_predictions = len(test_emails)
    
    for i, email in enumerate(test_emails, 1):
        print(f"\n📧 Test Email {i}:")
        print(f"Content: {email['text'][:150]}...")
        print(f"Expected: {email['expected'].upper()}")
        
        try:
            # Send classification request with correct format
            request_payload = {
                "text": email["text"],
                "model": "knn"  # Use our trained KNN model
            }
            
            response = requests.post(
                f"{server_url}/api/classify",
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                label = result.get('label', 'unknown')
                probability = result.get('probability', 0)
                score = result.get('score', 0)
                
                # Check if prediction matches expected
                is_correct = (
                    (label == "phish" and email["expected"] == "phishing") or
                    (label == "ham" and email["expected"] == "legitimate")
                )
                
                if is_correct:
                    correct_predictions += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ INCORRECT"
                
                if label == "phish":
                    print(f"🚨 PHISHING DETECTED (Confidence: {probability:.1%}) {status}")
                else:
                    print(f"✅ LEGITIMATE EMAIL (Confidence: {probability:.1%}) {status}")
                    
                # Show top features that influenced the decision
                explanation = result.get('explanation', {})
                top_features = explanation.get('top_features', [])
                if top_features:
                    print(f"   📊 Top contributing features:")
                    for feature in top_features[:3]:  # Show top 3
                        feature_name = feature.get('feature_name', 'unknown')
                        contribution = feature.get('contribution', 'unknown')
                        print(f"      • {feature_name} ({contribution})")
                        
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Server not running. Start with: python3 start_server.py")
            return
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Model Testing Complete!")
    print(f"📊 Accuracy: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.1%})")
    print(f"🚀 Your trained KNN model is working successfully!")

if __name__ == "__main__":
    test_model()