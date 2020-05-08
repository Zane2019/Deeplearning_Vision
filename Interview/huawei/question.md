

- > 性格测试：尽量积极阳光一些，不怕吃苦加班.

### 1.判断一棵树是否是二叉搜索树。
给定一个二叉树，判断其是否是一个有效的二叉搜索树。
假设一个二叉搜索树具有如下特征：节点的左子树只包含小于当前节点的数。节点的右子树只包含大于当前节点的数。所有左子树和右子树自身必须也是二叉搜索树。
**基本思路：解法一，深搜，递归传入子树的最大值和最小值;解法二，中序遍历有序**
```cpp
// class Solution {
// public:
//     const long int INT_MAX_=std::numeric_limits<long int>::max();
    
//     const long int INT_MIN_=std::numeric_limits<long int>::min();
 
//     bool isValidBST(TreeNode* root) {
//         return dfs(root,INT_MIN_,INT_MAX_);
        
//     }
//     bool dfs(TreeNode* root,long int min_,long int max_){
//         if(!root)return true;
//         if(root->val<=min_||root->val>=max_)
//             return false;
//         return dfs(root->left,min_,min(static_cast<long int>(root->val),max_))&&
//                 dfs(root->right,max(static_cast<long int>(root->val),min_),max_);
//     }
// };

class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if(!root) return true;
        stack<TreeNode*> s;
        TreeNode* pre=nullptr;
        TreeNode* curr=root;
        while(curr||!s.empty()){
            while(curr){
                s.push(curr);
                curr=curr->left;
            }
            curr=s.top();
            s.pop();
            if(pre&&pre->val>=curr->val)return false;
            pre=curr;
            curr=curr->right;
        }
        return true;
    }


};
```

### 2. 平凡根的函数实现

考虑牛顿迭代法:
$$x_{n+1}=x_n-f(x_n)/f^{'}(x_n)$$

### 3. 在O(nlogn)的前提下进行单链表排序


### 4.约瑟夫(Josephus)环问题
已知n个人(以编号1，2，3…n分别表示)围坐在一张圆桌周围。从编号为k的人开始报数，数到m的那个人出列;他的下一个人又从1开始报数，数到m的那个人又出列;依此规律重复下去，直到圆桌周围的人全部出列。通常解决这类问题时我们把编号从0~n-1，最后结果+1即为原问题的解
可以有多种解法,模拟解法,循环链表解法,递归解法.
递归解法记住公式:
$$f(N,M)=(f(N-1,M)+M)%N$$
f(N,M)表示n个人,报m就出列最终的胜利者.