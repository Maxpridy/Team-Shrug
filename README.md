# [카카오 아레나] 쇼핑몰 상품 카테고리 분류  

¯\\\_(ツ)\_/¯ 팀의 [카카오 아레나] 쇼핑몰 상품 카테고리 분류 코드입니다.

## 팀원
- [Maxpridy](https://github.com/Maxpridy)
- [ksw9446](https://github.com/ksw9446)


## 필요사항

- python 3.6 
- tensorflow >= 1.12


## 실행 방법
config의 경로를 위치에 맞게 설정한다.


### 1. train
```bash
python train.py
```


### 2. inference
저장 경로는 기본적으로 상위 폴더(../)이며 이 폴더에 "UmsoImg_v2_e00"와 같은 이름으로 저장된다.  

```bash
python inference.py  # default UmsoImg_v2_e31
python inference.py --path UmsoImg_v2_e15  # example
```
  

## 학습된 모델 링크
https://drive.google.com/file/d/1ltPPALttaP5s_R4_gVbO4ByOpl2-gzfi/view?usp=sharing  
위 모델의 크기는 182MB 입니다.


## Reference
1. [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181)
2. https://github.com/mkroutikov/tf-lstm-char-cnn


## Link
1. [Kakao Arena](https://arena.kakao.com/)  
2. [Maxpridy의 카카오 아레나를 마치고 느낀점](https://blog.naver.com/dustashy/221437058049)
