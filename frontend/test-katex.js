import {marked} from 'marked';
import markedKatex from 'marked-katex-extension';

marked.use(markedKatex({ 
  throwOnError: false
}));

marked.use({ gfm: true, breaks: false });

// 预处理函数 - 确保 $$ 块前后有空行
function preprocessMathBlocks(text) {
  let result = text;
  
  // Fix inline math followed by ( without space
  result = result.replace(/\$([^$\n]+)\$\(/g, '$$$$1$$ (');
  
  // 处理 $$ 块：确保前后有空行
  // 使用更精确的匹配：找到完整的 $$ 块并处理
  result = result.replace(/(\n)(\$\$)\n([\s\S]*?)\n(\$\$)(\n|$)/g, (match, before, open, content, close, after) => {
    // 确保开头 $$ 前有空行
    const prefix = '\n\n';
    // 确保结尾 $$ 后有空行（如果后面还有内容）
    const suffix = after === '\n' ? '\n\n' : '\n';
    return prefix + open + '\n' + content.trim() + '\n' + close + suffix;
  });
  
  return result;
}

// 测试：和实际文件一样的格式
const markdown = `- $D^1$: 3正2反
- $D^2$: 2正3反

基尼指数：
$$
\\text{Gini\\_index}(D, a) = \\frac{5}{10} \\times 0.48 + \\frac{5}{10} \\times 0.48 = 0.48
$$
`;

console.log('=== Original markdown ===');
console.log(JSON.stringify(markdown));
console.log('=== Preprocessed markdown ===');
const processed = preprocessMathBlocks(markdown);
console.log(JSON.stringify(processed));
console.log('=== HTML Output ===');
const result = await marked.parse(processed);
console.log(result);
console.log('=== End ===');
