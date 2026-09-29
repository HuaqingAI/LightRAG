import { describe, expect, test } from 'bun:test'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import DeleteDocumentsDialog from './DeleteDocumentsDialog'
import en from '@/locales/en.json'
import { renderWithProviders } from '@/test/render'

const strings = en.documentPanel.deleteDocuments

describe('DeleteDocumentsDialog defaults', () => {
  test('checks file and LLM cache deletion whenever the dialog opens', async () => {
    const user = userEvent.setup()
    renderWithProviders(<DeleteDocumentsDialog selectedDocIds={['doc-1']} />)

    const openButton = screen.getByRole('button', { name: strings.button })
    await user.click(openButton)

    const fileCheckbox = await screen.findByRole('checkbox', { name: strings.deleteFileOption })
    const cacheCheckbox = screen.getByRole('checkbox', { name: strings.deleteLLMCacheOption })
    expect(fileCheckbox).toBeChecked()
    expect(cacheCheckbox).toBeChecked()

    await user.click(fileCheckbox)
    await user.click(cacheCheckbox)
    await user.click(screen.getByRole('button', { name: en.common.cancel }))
    await user.click(openButton)

    expect(await screen.findByRole('checkbox', { name: strings.deleteFileOption })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: strings.deleteLLMCacheOption })).toBeChecked()
  })
})
