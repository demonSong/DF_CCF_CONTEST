package com.demon.utils;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLSentence;
import com.hankcs.hanlp.corpus.dependency.CoNll.CoNLLWord;

public class DemoDependencyParser {
	
	
	public static void main(String[] args) {
		CoNLLSentence sentence = HanLP.parseDependency("你这人怎么可以这样，我有什么缺点你直接说出来，别拐弯抹角");
		System.out.println(sentence);
		// 可以方便地遍历它
		for (CoNLLWord word : sentence) {
			System.out.printf("%s --(%s)--> %s\n", word.LEMMA, word.DEPREL, word.HEAD.LEMMA);
		}
		// 也可以直接拿到数组，任意顺序或逆序遍历
		CoNLLWord[] wordArray = sentence.getWordArray();
		for (int i = wordArray.length - 1; i >= 0; i--) {
			CoNLLWord word = wordArray[i];
			System.out.printf("%s --(%s)--> %s\n", word.LEMMA, word.DEPREL, word.HEAD.LEMMA);
		}
		// 还可以直接遍历子树，从某棵子树的某个节点一路遍历到虚根
		CoNLLWord head = wordArray[12];
		while ((head = head.HEAD) != null) {
			if (head == CoNLLWord.ROOT)
				System.out.println(head.LEMMA);
			else
				System.out.printf("%s --(%s)--> ", head.LEMMA, head.DEPREL);
		}
	}
}