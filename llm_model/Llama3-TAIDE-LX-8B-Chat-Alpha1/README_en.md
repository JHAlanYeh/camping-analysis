---
{}
---
* [中文版](./README.md)

# Introduction
* The [TAIDE project](https://taide.tw/index) aims to develop a generative AI dialogue engine model that is tailored to the linguistic and cultural characteristics of Taiwan, while also establishing a trustworthy AI environment. By combining academic, industrial, and research resources, the project seeks to advance the development of trustworthy generative AI, enhancing Taiwan's international competitiveness, promoting industrial development, and reducing dependence on foreign technologies.
* The Llama3 TAIDE series models are based on Meta's released [LLaMA3-8b model](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/), incorporating text and training materials from various fields in Taiwan to enhance the model's ability to respond in Traditional Chinese and perform specific tasks. The publicly released models are as follows:
    * [Llama3-TAIDE-LX-8B-Chat-Alpha1](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1): Based on LLaMA3-8b, continuous pretrained using Traditional Chinese data, and enhanced for office tasks and multi-turn dialogue capabilities through instruction tuning. Suitable for scenarios involving chat dialogue or task assistance. Llama3-TAIDE-LX-8B-Chat-Alpha1 also provides a [4-bit quantization model](https://huggingface.co/taide/Llama3-TAIDE-LX-8B-Chat-Alpha1-4bit). The quantization model is primarily offered for user convenience but may affect performance and introduce unforeseen issues. Users are advised to understand and take note of this.
  
# Model Parameters
* Parameters: 8B
* Max context length: 8K
* Training token in Traditional Chinese: 43B
* Training time: 2336 H100 GPU Hours

# Features
* Strictly vetting training data for the model to improve its trustworthiness and applicability.
* Enhancing the model's performance for tasks such as summarization, writing articles, writing letters, and translating between Chinese and English, which are commonly used in office settings.
* Strengthening the model's understanding of Taiwan-specific culture, terminology, and context.
* Equip the model with multi-turn conversation capabilities.

# Examples
| Task | User input | Model output |
| --- | -------- | ------- |
| Writing articles | 請以以下內容為基礎，寫一篇文章：撰寫一篇作文，題目為《一張舊照片》，內容要求為：選擇一張令你印象深刻的照片，說明令你印象深刻的原因，並描述照片中的影像及背後的故事。記錄成長的過程、與他人的情景、環境變遷和美麗的景色。 | 一張舊照片，承載著時光的痕跡和回憶的溫度。我還記得那張照片是怎麼出現的，那是一張泛黃的黑白相片，裱在牆上，訴說著一段屬於我家族的故事。<br><br>那張照片中，我的祖父母站在田野間，後方是綿延的山巒，前方則是一片金黃的稻田。這幅景象令我印象深刻，因為這是我的童年回憶，也是我成長過程中最美好的風景線。<br><br>我出生在台灣鄉下，從小就生活在這樣一個純樸的環境中。每當我跟著家人在田裡工作時，就會看到這樣的美景。金黃的稻穗隨著微風搖曳，山巒則靜靜地矗立在遠方，給人一種安詳與祥和的感覺。<br><br>那張照片拍攝於民國50年代，當時台灣經濟尚未起飛，農業為主要產業。我的祖父母都是農民，他們辛勤耕種，為我們家提供溫飽。每當我看到那張照片，就會想起他們為我們所做的犧牲和努力。<br><br>除了美麗的風景外，照片中還有一段動人的故事。在我還很小的時候，家裡發生了一場大火，將我們的房子和所有財產都燒光了。幸好全家人都平安無事，但生活卻變得十分困苦。我們被迫搬到另一個村莊，在那裡重新開始生活。<br><br>那張照片也記錄了那段艱辛的歲月。雖然我們失去了許多東西，但家人之間的感情卻更加深厚。我們互相扶持，共同面對生活上的種種挑戰。<br><br>如今，我已長大成人，在都市裡工作。但每當我看到那張舊照片，我都會想起那段簡單而美好的時光。那些美麗的風景、親人的笑容，以及那段充滿挑戰卻又充滿希望的日子，都將永遠鐫刻在我的記憶深處。<br><br>這張舊照片，不僅僅是一張照片，更是一個寶貴的家族史，一個關於我們如何克服困難、珍惜現在的故事。我將把這張照片好好保存下來，讓後代也能夠認識我們的根源，並且銘記那些值得珍藏的美好回憶。 |
| Writing letters   | 請以品牌經理的身份，給廣告公司的創意總監寫一封信，提出對於新產品廣告宣傳活動的創意建議。 | [您的姓名]<br>[您的職稱]<br>[您的公司名稱]<br>[郵件地址]<br>[電話號碼]<br>[發文日期]<br><br>[創意總監姓名]<br>[廣告公司名稱]<br>[廣告公司地址]<br><br>主旨:關於新產品宣傳活動創意發想，請查照。<br><br>尊敬的[創意總監姓名]，<br><br>我希望這封信能找到你忙碌中好好休息。我寫這封信，是為了分享我對於我們即將推出的新產品宣傳活動一些創意發想。我深信，這些想法能幫助我們新產品在市場上脫穎而出，吸引消費者的目光。<br><br>首先，我建議我們可以利用社群媒體平台，像是Facebook、Instagram、YouTube等，製作一系列引人入勝的短片和貼文。這些內容可以展現新產品的特色和優點，同時也能傳達出我們品牌重視環保、健康的核心價值。我們可以與網紅或意見領袖合作，讓他們試用並分享他們的使用經驗，這將有助於增加我們產品的可信度和能見度。<br><br>其次，我們可以舉辦新產品試吃/試用活動，邀請民眾親身體驗我們的新產品。我們可以在各大百貨公司、超市、以及我們的門市舉辦這些活動，讓參與者免費試吃/試用新產品，並填寫問卷回饋意見。這種互動的方式，不僅能收集到寶貴的市場反饋，還能讓民眾對我們的品牌產生正面的印象。<br><br>第三，我們可以與當地餐飲業者合作，將我們的新產品納入他們的菜單中。這將使我們的產品接觸到更廣泛的客群，同時也能證明我們的產品在烹飪和美食產業中具有價值。<br><br>第四，我們可以舉辦一場新產品發表會，邀請媒體、業界人士、以及重要客戶參加。在這場活動中，我們可以展示新產品的各項特色和優勢，並由專業廚師示範如何將新產品融入各式料理中。這將是向大眾展示我們品牌創新精神和專業能力的絕佳機會。<br><br>最後，我建議我們可以製作一系列精美的包裝和禮盒，以吸引消費者在節慶和送禮時選擇我們的產品。精緻的包裝將能提升我們產品的價值感，同時也能傳達出我們品牌對細節和品質的堅持。<br><br>我深信，這些創意發想將有助於我們新產品在市場上獲得成功。若您對這些想法有任何意見或建議，請不吝與我分享。我期待著與您合作，共同為[您的公司名稱]的成長和成功而努力。<br><br>謝謝您抽空閱讀我的信，並期待著與您進一步討論這些想法。<br><br>敬禮<br><br>[您的姓名]<br>[您的職稱]<br>[您的公司名稱] |
| Summarization   | 請將這篇文章精簡條理化:「產業創新條例第10條之2及第72條條文修正案」俗稱「台版晶片法」,針對半導體、電動車、5G等技術創新且居國際供應鏈關鍵地位公司,提供最高25%營所稅投抵優惠,企業適用要件包含當年度研發費用、研發密度達一定規模,且有效稅率達一定比率。<br>為因應經濟合作暨發展組織(OECD)國家最低稅負制調整,其中有效稅率門檻,民國112年訂為12%,113年料將提高至15%,但仍得審酌國際間最低稅負制實施情形。<br>經濟部官員表示,已和財政部協商進入最後階段,除企業研發密度訂在6%,目前已確認,企業購置先進製程的設備投資金額達100億元以上可抵減。<br>財政部官員表示,研商過程中,針對台灣產業與其在國際間類似的公司進行深入研究,在設備部分,畢竟適用產創10之2的業者是代表台灣隊打「國際盃」,投入金額不達100億元,可能也打不了。<br>至於備受關注的研發費用門檻,經濟部官員表示,歷經與財政部來回密切討論,研發費用門檻有望落在60億至70億元之間。<br>財政部官員指出,研發攸關台灣未來經濟成長動能,門檻不能「高不可攀」,起初雖設定在100億元,之所以會調降,正是盼讓企業覺得有辦法達得到門檻、進而適用租稅優惠,才有動力繼續投入研發,維持國際供應鏈關鍵地位。<br>經濟部官員表示,因廠商研發費用平均為30、40億元,其中,IC設計業者介於30億至60億元範圍,若將門檻訂在100億元,符合條件的業者較少、刺激誘因不足;此外,若符合申請門檻的業者增加,將可提高企業在台投資金額,財政部稅收也能因此獲得挹注。<br>IC設計業者近日頻頻針對產創10之2發聲,希望降低適用門檻,加上各國力拚供應鏈自主化、加碼補助半導體產業,經濟部官員表示,經濟部和財政部就產創10之2達成共識,爭取讓更多業者受惠,盼增強企業投資力道及鞏固台灣技術地位。<br>財政部官員表示,租稅獎勵的制定必須「有為有守」,並以達到獎勵設置目的為最高原則,現階段在打「國內盃」的企業仍可適用產創第10條、10之1的租稅優惠,共同壯大台灣經濟發展。<br>經濟部和財政部正就研發費用門檻做最後確認,待今明兩天預告子法之後,約有30天時間,可與業界進一步討論及調整,盼產創10之2能在6月上路。 | 文章重點摘要如下:<br><br>- 「台版晶片法」或《產業創新條例第10條之2及第72條條文修正案》，旨在鼓勵關鍵技術創新企業，特別是那些在半導體、電動車、5G技術等領域且在國際供應鏈中扮演關鍵角色者。<br><br>- 企業若符合特定條件，包括當年度研發費用及研發密度達標，可享最高25%的營所稅抵減優惠。研發密度目標設為6%，而企業為取得抵減，必須證明其有效稅率達到特定比率，起初設定為112年的12%，113年提高至15%，但可能會依國際稅制的實施情形進行調整。<br><br>- 政府在制定政策時，考量到OECD最低稅負制的調整，因此會視情況調整有效稅率門檻。<br><br>- 政府部門已就法案內容與企業利益相關者進行協商，確定一些細節，例如企業投資先進製程設備的門檻設為100億元新台幣以供抵減。<br><br>- 研發費用門檻經過討論，可能從最初提議的100億元下調至60億至70億元之間，以激勵更多企業符合資格並投入研發，維持台灣在國際供應鏈中的關鍵地位。<br><br>- 經濟部官員表示，調降門檻的目的是讓更多企業覺得能夠達成標準，從而獲得應用租稅優惠的動力，並可能藉此提高在台投資金額，進而增加政府稅收。<br><br>- 由於IC設計業者的研發費用普遍較低，若門檻設得過高，符合條件的企業將很少，因此有必要調整。<br><br>- 在各國推動供應鏈自主化並增加對半導體產業的補助之際，經濟部正努力確保更多企業能從《產業創新條例》第10條之2中受益，以增強投資和鞏固台灣的技術地位。<br><br>- 政府強調租稅優惠的制定應符合獎勵目標，在此同時，也將繼續提供其他租稅優惠給在國內經營的企業，以促進台灣經濟成長。<br><br>- 最終的研發費用門檻將由經濟部和財政部共同確定，在預告相關子法後，將有約30天的時間供各界討論並提出調整建議，目標是讓這項新政策在6月上路。 |
| Translation   | 以下提供英文內容，請幫我翻譯成中文。Dongshan coffee is famous for its unique position, and the constant refinement of production methods. The flavor is admired by many caffeine afficionados. | 東山咖啡以獨特的位置，加上不斷精進的製作手法而聞名，風味更是讓許多咖啡癡趨之若鶩。 |

# Usage
* Sample code
    * The sample code and documentation will be released on GitHub later.

* Prompt template
    * Normal QA
        ```python
        chat = [
            {"role": "user", "content": "{question}"},
        ]
        prompt = tokenizer.apply_chat_template(chat)
        ```
        * Replace {question} with user input
    * QA with system prompt
        ```python
        chat = [
            {"role": "system", "content": "{sys}"},
            {"role": "user", "content": "{question}"},
        ]
        prompt = tokenizer.apply_chat_template(chat)
        ```
        * Replace {sys} with system prompt，ex：你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。
        * Replace {question} as user input
    * Multi turns conversation
        ```python
        chat = [
            {"role": "system", "content": "{sys}"},
            {"role": "user", "content": "{question1}"},
            {"role": "assistant", "content": "{model_anwer_1}"},
            {"role": "user", "content": "{question2}"},
        ]
        prompt = tokenizer.apply_chat_template(chat)
        ```
        * Replace {sys} with system prompt，ex：你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。
        * Replace {question1} with user input 1
        * Replace {model_anwer_1} with model response 1
        * Replace {question2} with user input 2
    * For more details, please refer to the [Llama 3 documentation](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)

# Training methods
* Software / hardware spec
    * GPU: H100
    * Training Framework: PyTorch
* Data preprocessing
    * Character normalization
    * Deduplication
    * Denoise
        * Html tag、javascript in web content
        * Non-standard characters or garbage characters
        * Posts with an insufficient number of characters
        * Removing specific formats such as extra line breaks added for formatting purposes
    * Removing personal information such as emails and phone numbers.
    * Remove inappropriate content such as gambling, pornography, etc..
* Continuous pretraining (CP)
    * Supplementing the model with a large amount of reliable Traditional Chinese knowledge.
    * Hyper parameters
        * optimizer: AdamW
        * learning rate: 1e-4
        * batch size: 1M tokens
        * epoch: 1
* Fine tune (FT)
    * Enabling the model to answer questions in Traditional Chinese.
    * Hyper parameters
        * optimizer: AdamW
        * learning rate: 5e-5
        * batch size: 256K tokens
        * epoch: 3

# Training Data
* Continuous pre-training data (about 140GB)
| Dataset | Description |
| --- | -------- |
| Litigation Data | [Civil litigation data](https://judgment.judicial.gov.tw/FJUD/default.aspx) from various levels of courts in the judicial rulings, including data from 2013/01 to 2023/12. |
| CNA news | The [CNA news](https://www.cna.com.tw/) includes daily news articles from June 1993 to June 2023, spanning a period of 30 years. The content covers various domains such as domestic and international politics, society, economy, culture, education, and lifestyle. |
| ETtoday news | [ETtoday news](https://www.ettoday.net/) data, including data from 2011/10 to 2023/12. |
| Legislative Yuan Gazette | The [Legislative Yuan Gazette](https://ppg.ly.gov.tw/ppg/) contains data from the 1st session of the 8th term to the 7th session of the 10th term. |
| Publisher Website Book Introduction | Includes book introduction data from the websites of [SunColor](https://www.suncolor.com.tw/), [Gotop](https://www.gotop.com.tw/) publishers. |
| Abstracts of GRB research projects | [GRB](https://www.grb.gov.tw/) is an information system that compiles research projects funded by government grants and their outcome reports. This dataset primarily includes research project abstracts from 1993 to 2023, including both Chinese and their English counterparts. |
| Academic conference proceedings abstracts | The [database](https://sticnet.stpi.narl.org.tw/sticloc/ttscalle?meet:) contains academic conference proceedings held in Taiwan from 1988 to 2009. |
| Taiwan Panorama magazine | [Taiwan Panorama magazine](https://www.taiwan-panorama.com/) contains articles from July 1993 to June 2023, spanning 30 years. The content focuses on Taiwanese culture, tourism, and local customs. |
| 樂詞網 | 《[樂詞網](https://terms.naer.edu.tw/)》covers approximately 187,000 academic terms in the humanities and social sciences, along with their translations. |
| Data from various ministries and commissions | Including partial data from government department websites such as the Executive Yuan's "[National Overview](https://www.ey.gov.tw/state/)", the Ministry of Culture's "[National Cultural Memory Bank](https://memory.culture.tw/)", the National Development Council's "[Archives Support Teaching Network](https://art.archives.gov.tw/index.aspx)", the Ministry of Transportation's "[Traffic Safety Portal](https://168.motc.gov.tw/)", etc. |
| Business Today | [Business Today](https://www.businesstoday.com.tw/) Magazine is a weekly magazine focused on finance. The dataset includes articles from 2008/01 to 2023/07. |
| Mandarin and idiom dictionary from the Ministry of Education | Dataset including:<br>[Idiom Dictionary](https://dict.idioms.moe.edu.tw/search.jsp?webMd=1&la=0): Contains 5,338 idioms, including definitions, original stories, usage explanations, and example sentences.<br>[Revised Mandarin Dictionary](https://dict.revised.moe.edu.tw/?la=0&powerMode=0): contains Chinese words and various vocabulary, including pronunciation, radicals, definitions, and other information, totaling approximately 165,539 entries.<br>[Concise Mandarin Dictionary](https://dict.concised.moe.edu.tw/?la=0&powerMode=0): is a condensed version of the "Revised Mandarin Dictionary", containing a total of 45,247 entries. |
| SCITechVista | The dataset includes science news and popular science articles from the [SCITechVista](https://scitechvista.nat.gov.tw/) website. |
| iKnow | The [iKnow](https://iknow.stpi.narl.org.tw/) platform provides information on market trends, strategic analysis, patent knowledge, and technology transaction information for Taiwan and the global technology industry. The dataset includes data from 2005/01 to 2023/07. |
| Science Development Monthly Magazine | [Science Development Monthly Magazine](https://ejournal.stpi.narl.org.tw/sd) is a popular science publication published by the National Science Council (NSC) to promote science education. It includes articles from 2004/10 to 2020/12. In 2021, the magazine was relaunched as "[CharmingSCITech](https://www.charmingscitech.nat.gov.tw/)" quarterly, providing new knowledge on international technology issues. |
| Legislation Database | The [Legislation Database](https://law.moj.gov.tw/) includes the latest central regulations, rules, draft bills, and local regulations issued by government agencies as of 2023/10. |
| Local Government Tourism Websites | Covering partial data from tourism websites of local government counties and cities in Taiwan. |
| Curriculum Guidelines from the National Institute of Education | The dataset includes curriculum guidelines for different subjects at various levels of education. |
| CNA's English and Chinese Name Translation Database | The English and Chinese Name Translation Database of the Central News Agency (CNA) collects translations of foreign and Chinese surnames, personal names, organizations, and place names used in news. |
| Fairy tales | A total of 20 fairy tale books, including "Tom Sawyer," "Peter Pan," "Alice's Adventures in Wonderland," "Uncle Long Legs," and more. |
| RedPajama-Data-V2 | Extracting English data from the [RedPajama-Data-v2](https://github.com/togethercomputer/RedPajama-Data) multilingual dataset |
| MathPile-commercial | A mathematics-focused dataset obtained from [MathPile-commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial) |
| Traditional Chinese Wikipedia Articles | The content of all articles in [Traditional Chinese Wikipedia](https://zh.wikipedia.org/zh-tw/%E4%B8%AD%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91), up to January 2023. |
| github-code-clean | An open-source code dataset on GitHub. After removing unlicensed code and documents. |
* Fine tune data
    * The TAIDE team trains the LLaMA2 series models to generate fine-tuning data, which generates single or multi-turn conversations on topics such as world knowledge, creative writing, general knowledge, translation, summarization, programming, and Taiwanese values. The fine tune data consists of 128K prompt-response pairs and will be released publicly later.

# Evaluation
* taide-bench
    * Data
        * Tasks include writing articles, writing letters, summarizing articles, translating from English to Traditional Chinese, translating from Traditional Chinese to English. There are 500 questions in total.
        * data link: [taide-bench](https://huggingface.co/datasets/taide/taide-bench)
    * Evaluation method
        * LLM as a Judge by GPT4
        * code link: [taide-bench-eval](https://github.com/taide-taiwan/taide-bench-eval)
    * Scores
| Model | Translating from Traditional Chinese to English | Translating from English to Traditional Chinese | Summerization | Writing articles | Writing letters | Average |
| --- | ----- | ----- | ---- | ---- | ---- | --- |
| Llama3-TAIDE-LX-8B-Chat-Alpha1 | 7.770 | 8.280 | 8.495 | 9.605 | 8.950 | 8.620 |
| GPT3.5 | 8.880 | 8.810 | 7.450 | 9.490 | 8.750 | 8.676 |
| TAIDE-LX-7B-Chat | 7.165 | 7.685 | 7.720 | 9.635 | 9.110 | 8.263 |
| LLAMA2 7B | 6.075 | 4.475 | 5.905 | 2.625 | 3.040 | 4.424 |
| LLAMA2 13B | 6.480 | 6.135 | 6.110 | 2.565 | 3.000 | 4.858 |
| LLAMA2 70B | 6.975 | 6.375 | 6.795 | 2.625 | 2.990 | 5.152 |

# License
* [Llama3-TAIDE Models Community License Agreement](https://drive.google.com/file/d/12-Q0WWSjG0DW6CqJQm_jr5wUGRLeb-8p/view)

# Disclaimer
* Due to limitations in its design architecture and the inevitable biases in data, any response from the LLM model does not represent the stance of TAIDE. Additional security measures should be implemented before use, and responses may also contain incorrect information. Users are advised not to fully trust the responses.

# Development Team
* [https://taide.tw/index/teamList](https://taide.tw/index/teamList)

# Useful links
* [TAIDE official website](https://taide.tw/index)
* [TAIDE Huggingface](https://huggingface.co/taide)
* [TAIDE Github](https://github.com/taide-taiwan)
* [Kuwa AI](https://kuwaai.org/)

# Citation
* [TAIDE official website](https://taide.tw/index)
