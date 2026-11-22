<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import OreArt from '../components/OreArt.vue'

const THEMES = [
  {
    word: '方解石',
    en: 'CALCITE',
    emoji: '💎',
    color: '#f1d3a4',
    color2: '#f7ead6',
    ink: '#4a3a24',
    desc: '碳酸钙晶体，透明时像冰块，遇盐酸会咕嘟咕嘟冒泡。',
    variant: 0
  },
  {
    word: '石英',
    en: 'QUARTZ',
    emoji: '🔮',
    color: '#cbb8ef',
    color2: '#efe9fb',
    ink: '#3b3058',
    desc: '地壳里最常见的矿物之一，水晶、玛瑙、玉髓都是它的家人。',
    variant: 1
  },
  {
    word: '赤铁矿',
    en: 'HEMATITE',
    emoji: '🪨',
    color: '#d2917c',
    color2: '#f6ded4',
    ink: '#4d2a1f',
    desc: '红褐色的重要铁矿石，条痕是樱红色，名字就来自它的颜色。',
    variant: 2
  },
  {
    word: '黄铁矿',
    en: 'PYRITE',
    emoji: '✨',
    color: '#e8c569',
    color2: '#fbf0cf',
    ink: '#4a370d',
    desc: '金光闪闪的“愚人金”，条痕却是绿黑色，别被它骗啦。',
    variant: 3
  }
]

const index = ref(0)
const theme = computed(() => THEMES[index.value])

const heroStyle = computed(() => ({
  background: `linear-gradient(135deg, ${theme.value.color2} 0%, ${theme.value.color} 55%, ${theme.value.color2} 100%)`
}))

let timer = setInterval(() => {
  index.value = (index.value + 1) % THEMES.length
}, 5000)

function setTheme(i) {
  index.value = i
  clearInterval(timer)
  timer = setInterval(() => {
    index.value = (index.value + 1) % THEMES.length
  }, 5000)
}

onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
  <PageHeader />

  <section class="hero" :style="heroStyle">
    <div class="hero-bg" :style="heroStyle" />
    <div class="hero-grain" />

    <div class="hero-word" :style="{ color: theme.color2 }">{{ theme.word }}</div>

    <div class="hero-content">
      <div class="hero-copy" :style="{ color: theme.ink }">
        <span class="hero-kicker">识矿乐园 · MINE WONDERLAND</span>
        <h1 class="hero-title">
          认识脚下的<br />每一块石头
          <span>{{ theme.word }} · {{ theme.en }}</span>
        </h1>
        <p class="hero-desc">{{ theme.desc }}</p>
        <div class="hero-actions">
          <router-link to="/shoot" class="btn btn-primary">📷 拍一拍，识别矿石</router-link>
          <router-link to="/gallery" class="btn btn-ghost" :style="{ borderColor: theme.ink }">
            🗂️ 打开矿石图鉴
          </router-link>
        </div>
      </div>

      <div class="hero-stage">
        <OreArt :color="theme.color" :variant="theme.variant" />
        <OreArt color="#ffffff" :variant="theme.variant + 1" />
        <OreArt :color="theme.color" :variant="theme.variant + 2" />
        <OreArt color="#f6e3b4" :variant="theme.variant + 3" />
      </div>
    </div>

    <div class="hero-dots">
      <button
        v-for="(item, i) in THEMES"
        :key="item.en"
        :class="{ active: i === index }"
        :aria-label="item.word"
        @click="setTheme(i)"
      />
    </div>
  </section>

  <section class="hero-features">
    <article class="feature">
      <div class="icon">📷</div>
      <h3>拍照识别</h3>
      <p>上传一张矿石照片，模型会给出类别、置信度和 Top-3 结果，还会念给你听。</p>
    </article>
    <article class="feature">
      <div class="icon">📖</div>
      <h3>知识图鉴</h3>
      <p>方解石、石英、赤铁矿、长石、黄铁矿，每识别一种就点亮一张知识卡。</p>
    </article>
    <article class="feature">
      <div class="icon">📈</div>
      <h3>记录成长</h3>
      <p>登录后保存识别记录、查看七日统计和类别分布，还能提交纠错帮助模型进步。</p>
    </article>
  </section>
</template>
