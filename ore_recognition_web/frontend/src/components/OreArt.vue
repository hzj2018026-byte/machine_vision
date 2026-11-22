<script setup>
import { computed } from 'vue'

const props = defineProps({
  color: { type: String, default: '#2f8f83' },
  variant: { type: Number, default: 0 }
})

const rotate = computed(() => [-6, 8, -3, 5, 0][props.variant % 5])
const gradientId = `ore-shine-${Math.random().toString(36).slice(2, 9)}`
</script>

<template>
  <svg
    class="ore-art"
    viewBox="0 0 200 220"
    role="img"
    aria-label="矿石水晶插画"
    :style="{ transform: `rotate(${rotate}deg)` }"
  >
    <defs>
      <linearGradient :id="gradientId" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="0.55" />
        <stop offset="55%" stop-color="#ffffff" stop-opacity="0.06" />
        <stop offset="100%" stop-color="#000000" stop-opacity="0.12" />
      </linearGradient>
    </defs>

    <!-- 底座阴影 -->
    <ellipse cx="101" cy="199" rx="68" ry="13" fill="#23302e" opacity="0.12" />

    <!-- 左侧小晶体 -->
    <polygon points="60,192 38,128 56,92 78,132 82,192" :fill="color" opacity="0.82" />
    <polygon points="56,92 63,132 48,160 38,128" fill="#ffffff" opacity="0.38" />
    <polygon points="56,92 78,132 63,132" fill="#000000" opacity="0.12" />

    <!-- 中间主晶体 -->
    <polygon points="100,194 72,92 96,34 126,86 130,194" :fill="color" />
    <polygon points="96,34 126,86 108,118 82,104" :fill="`url(#${gradientId})`" />
    <polygon points="96,34 82,104 72,92" fill="#ffffff" opacity="0.42" />
    <polygon points="126,86 108,118 130,194 118,190" fill="#000000" opacity="0.14" />
    <polygon points="72,92 100,112 108,118 82,150 78,132" fill="#ffffff" opacity="0.14" />

    <!-- 右侧晶体 -->
    <polygon points="132,192 130,122 150,82 168,128 164,192" :fill="color" opacity="0.9" />
    <polygon points="150,82 168,128 150,152 138,126" fill="#ffffff" opacity="0.34" />
    <polygon points="150,82 150,152 132,122" fill="#000000" opacity="0.1" />

    <!-- 顶部碎晶 -->
    <polygon points="86,58 78,34 92,16 102,40" :fill="color" opacity="0.75" />
    <polygon points="78,34 92,16 90,38" fill="#ffffff" opacity="0.4" />

    <!-- 高光点 -->
    <circle cx="148" cy="118" r="4" fill="#ffffff" opacity="0.7" />
    <circle cx="89" cy="128" r="3" fill="#ffffff" opacity="0.55" />
  </svg>
</template>
