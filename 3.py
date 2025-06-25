import re

with open('./monitor_directory/1.py', 'r', encoding='utf-8') as f:
    code = f.read()

print(code)
# 1. 여러 줄 주석: """ """ 또는 ''' '''
multiline_comments = re.findall(r'("""|\'\'\')[\s\S]*?\1', code)

# 2. 한 줄 주석: #
singleline_comments = re.findall(r'#.*', code)

# 3. SQL 문 탐지 (Select 문 위주, 대소문자 구분 없이)
sql_statements = re.findall(r'\bselect\b[\s\S]*?;', code, re.IGNORECASE)

# 4. 이메일 주소 탐지
emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', code)

# 5. 전체 주석 (단일 + 멀티라인 합치기)
all_comments = multiline_comments + singleline_comments

# 출력
print("🟡 주석 탐지 결과:")
for comment in all_comments:
    print(comment.strip())
    
print("\n🔵 SQL 문 탐지 결과:")
for sql in sql_statements:
    print(sql.strip())

print("\n🟣 이메일 주소 탐지 결과:")
for email in emails:
    print(email)