import { defaultTheme, defineUserConfig } from 'vuepress'
import { shikiPlugin } from '@vuepress/plugin-shiki'
import { gitPlugin } from '@vuepress/plugin-git'
import { readFileSync } from "fs"
import codeCopyPlugin from '@snippetors/vuepress-plugin-code-copy'

const mojoGrammar = JSON.parse(readFileSync("./syntax/mojo.tmLanguage.json").toString())

export default defineUserConfig({
    lang: 'en-US',
    title: 'fast.ai course',
    description: 'Practical Deep Learning for Coders exercises',
    pagePatterns: ['**/*.md', '!**/README.md', '!.vuepress', '!node_modules', '!venv'],
    markdown: {
        code: {
            lineNumbers: false
        }
    },
    theme: defaultTheme({
        colorMode: 'dark',
        logo: '/hero.png',
        repo: 'mojodojodev/fastai-course',
        repoLabel: 'GitHub',
        docsRepo: 'mojodojodev/fastai-course',
        docsBranch: 'main',
        lastUpdated: false,
        locales: {
            '/': {
                selectLanguageName: 'English',
                editLinkText: 'Edit this page on GitHub',
                sidebar: [
                    '01-exercise',
                ],
            }
        }
    }),
    plugins: [
        gitPlugin({
            contributors: false
        }),
        codeCopyPlugin(),
        shikiPlugin({
            langs: [
                {
                    id: "mojo",
                    scopeName: 'source.mojo',
                    grammar: mojoGrammar,
                    aliases: ["Mojo"],
                },
                {
                    id: "python",
                    scopeName: 'source.python',
                    path: "./languages/python.tmLanguage.json",
                    aliases: ["Python"]
                },
                {
                    id: "output",
                    path: "./languages/python.tmLanguage.json",
                    aliases: ["Output"],
                    scopeName: 'source.python',
                },
                {
                    id: "shell",
                    scopeName: 'source.shell',
                    path: "./languages/shellscript.tmLanguage.json",
                    aliases: ["bash", "Bash"]
                },
            ],
            theme: 'material-default',
        }),
    ],
});
