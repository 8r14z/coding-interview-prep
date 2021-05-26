MOD=10**9+7
def alph(s):return '#abcdefghijklmnopqrstuvwxyz'.index(s)
for case in range(1,int(input())+1):
    n,k=map(int,input().split())
    s=input()
    tot=0

    for i in range((n+1)//2):
        tot+=(alph(s[i])-1)*pow(k,(n+1)//2-i-1,MOD)
		
    if n%2==0 and s[:n//2]+s[n//2-1::-1]<s:tot+=1
    if n%2==1 and s[:(n+1)//2]+s[(n+1)//2-2::-1]<s:tot+=1
    print(f'Case #{case}: {tot%MOD}')



# res += (ord(S[i]) - 97)*(k**(mid-i-1))