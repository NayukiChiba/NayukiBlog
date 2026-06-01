import { visit } from 'unist-util-visit';

function getLabelText(node) {
  return node?.children
    ?.map((child) => (child.type === 'text' ? child.value : ''))
    .join('')
    .trim() || '';
}

export function remarkContainerDirectives() {
  return (tree) => {
    visit(tree, 'containerDirective', (node) => {
      const type = node.name;
      const labelNode = node.children?.[0];

      if (type === 'fold') {
        const title = getLabelText(labelNode);
        if (!labelNode?.data?.directiveLabel || !title) return;
        node.children.shift();
        node.data ??= {};
        node.data.hName = 'details';
        node.children.unshift({ type: 'html', value: `<summary>${title}</summary>` });
      }
    });
  };
}
