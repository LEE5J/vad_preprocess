# vad_compare_util.py
from collections import defaultdict

# 에러 메시지 수집용 클래스
class ErrorCollector:
    def __init__(self):
        self.errors = defaultdict(list)
    
    def add(self, category, message):
        self.errors[category].append(message)
    
    def print_all(self):
        if not self.errors:
            return
        
        print("\n===== 수집된 에러 메시지 =====")
        for category, messages in self.errors.items():
            print(f"\n{category} 관련 에러 ({len(messages)}개):")
            for i, msg in enumerate(messages[:20], 1):  # 카테고리별 최대 20개만 출력
                print(f"{i}. {msg}")
            
            if len(messages) > 20:
                print(f"... 외 {len(messages) - 20}개 더 있음")
        print("==============================\n")
    
    def clear(self):
        self.errors.clear()